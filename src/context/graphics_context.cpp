#include "context.h"
#include "swapchain.h"
#include "cpu_sim.h"
#include "gpu_sim.h"
#include "camera.h"
#include "vertex.h"
#include "texture_2d.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "gui.h"

#include "graphics_context.h"

GraphicsContext::GraphicsContext(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager) 
	: context_(context), swapchain_(swapchain), texture_manager_(textureManager), model_manager_(modelManager)
{
	//msaa_samples_ = GetMaxUsableSampleCount();

	CreateCommandBuffers();
	CreateQueryPool();

	CreateDescriptorSetLayout();
	CreateDescriptorPools();

	CreateUniformBuffers();

	CreateDescriptorSets();
	CreateGraphicsPipelines();
	CreateSyncObjects();

	CreateDepthResources();

	cpu_sim_ = std::make_unique<CpuSim>(context_, swapchain_, texture_manager_, graphics_.global_set_layout, Nx_, Ny_, spacing_);
	gpu_sim_ = std::make_unique<GpuSim>(context_, swapchain_, texture_manager_, graphics_.global_set_layout, Nx_, Ny_, spacing_);
}

GraphicsContext::~GraphicsContext()
{

}

void GraphicsContext::Update(Camera& camera)
{

	if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		cpu_sim_->SimulateClothXPBD_CPU(
			model_manager_.models[0]->position_, model_manager_.models[0]->radius_
		);
	}
	else if (cpu_or_gpu_ == CpuOrGpu::GPU)
	{
		gpu_sim_->UpdateComputeUBO(current_frame_, model_manager_.models[0]);
	}

	UpdateGraphicsUBO(camera);
}

void GraphicsContext::Draw(std::unique_ptr<GUI>& gui)
{
	auto [result, imageIndex] = swapchain_.swapchain_.acquireNextImage(UINT64_MAX, nullptr, in_flight_fences_[current_frame_]);

	while (vk::Result::eTimeout == context_.device_.waitForFences(*in_flight_fences_[current_frame_], vk::True, UINT64_MAX));
	context_.device_.resetFences(*in_flight_fences_[current_frame_]);

	uint64_t computeWaitValue;
	uint64_t computeSignalValue;
	uint64_t graphicsWaitValue;
	uint64_t graphicsSignalValue;

	if (cpu_or_gpu_ == CpuOrGpu::GPU)
	{
		computeWaitValue = timeline_value_;
		computeSignalValue = ++timeline_value_;
		graphicsWaitValue = computeSignalValue;
		graphicsSignalValue = ++timeline_value_;

		gpu_sim_->ComputeRecord(current_frame_, cmds_.compute[current_frame_], timestamp_pool_, timestampSteps, test_scene_);

		{
			// Submit compute work
			vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
				.waitSemaphoreValueCount = 1,
				.pWaitSemaphoreValues = &computeWaitValue,
				.signalSemaphoreValueCount = 1,
				.pSignalSemaphoreValues = &computeSignalValue
			};

			vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eComputeShader };

			vk::SubmitInfo computeSubmitInfo{
				.pNext = &computeTimelineInfo,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &*semaphore_,
				.pWaitDstStageMask = waitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &*cmds_.compute[current_frame_],
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &*semaphore_
			};

			context_.queue_.submit(computeSubmitInfo, nullptr);
		}

		if (gui->is_print_timestamps)
		{
			uint32_t prev = (current_frame_ + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;

			vk::SemaphoreWaitInfo waitInfo{
				.semaphoreCount = 1,
				.pSemaphores = &*semaphore_,
				.pValues = &computeSignalValue // 또는 computeSignalValue
			};
			while (vk::Result::eTimeout == context_.device_.waitSemaphores(waitInfo, UINT64_MAX));

			float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
			float toMs = nsPerTick / 1e6f;

			uint32_t numTimestamp = timestampSteps;
			std::vector<uint64_t> ts(numTimestamp);

			VkResult res = vkGetQueryPoolResults(
				static_cast<VkDevice>(*context_.device_),
				static_cast<VkQueryPool>(*timestamp_pool_),
				0, numTimestamp,
				ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
				VK_QUERY_RESULT_64_BIT
			);

			auto delta_ms = [&](uint32_t i0, uint32_t i1) {
				return (ts[i1] - ts[i0]) * toMs;
				};

			// 1) Integrate / Clear Lambdas
			float tIntegrate = delta_ms(0, 1);
			float tClearLambdas = delta_ms(2, 3);

			// 2) Iteration 안의 것들은 전부 합산
			float tSolveStretch = 0.0f;
			float tSolveDiag = 0.0f;
			float tSolveBend = 0.0f;
			float tApplyDeltas = 0.0f;
			float tCollideSdf = 0.0f;

			for (uint32_t it = 0; it < gpu_sim_->iterations_; ++it)
			{
				uint32_t base = 4 + it * gpu_sim_->iter_contraint_count_;
				tSolveStretch += delta_ms(base + 0, base + 1);
				tSolveDiag += delta_ms(base + 2, base + 3);
				tSolveBend += delta_ms(base + 4, base + 5);
				tApplyDeltas += delta_ms(base + 6, base + 7);
				tCollideSdf += delta_ms(base + 8, base + 9);
			}

			// 3) Update Velocity
			uint32_t lastBase = 4 + gpu_sim_->iter_contraint_count_ * gpu_sim_->iterations_;
			double tUpdate = delta_ms(lastBase + 0, lastBase + 1);

			// 4) 출력
			double total = tIntegrate + tClearLambdas + tSolveStretch +
				tSolveDiag + tSolveBend + tApplyDeltas + tCollideSdf + tUpdate;
			uint32_t c = 0;
			{
				c = 0;
				label_time_[labels_[c++]] = tIntegrate;
				label_time_[labels_[c++]] = tClearLambdas;
				label_time_[labels_[c++]] = tSolveStretch;
				label_time_[labels_[c++]] = tSolveDiag;
				label_time_[labels_[c++]] = tSolveBend;
				label_time_[labels_[c++]] = tApplyDeltas;
				label_time_[labels_[c++]] = tCollideSdf;
				label_time_[labels_[c++]] = tUpdate;
				label_time_[labels_[c++]] = total;
			}

			{
				c = 0;
				label_avg_time_[labels_[c++]] += tIntegrate;
				label_avg_time_[labels_[c++]] += tClearLambdas;
				label_avg_time_[labels_[c++]] += tSolveStretch;
				label_avg_time_[labels_[c++]] += tSolveDiag;
				label_avg_time_[labels_[c++]] += tSolveBend;
				label_avg_time_[labels_[c++]] += tApplyDeltas;
				label_avg_time_[labels_[c++]] += tCollideSdf;
				label_avg_time_[labels_[c++]] += tUpdate;
				label_avg_time_[labels_[c++]] += total;
			}

			time_count_++;

		}
	}
	else if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		graphicsWaitValue = timeline_value_;
		graphicsSignalValue = ++timeline_value_;
	}

	RecordGraphicsCommandBuffer(imageIndex);
	{
		vk::PipelineStageFlags graphicsWaitStage = vk::PipelineStageFlagBits::eVertexInput;
		vk::TimelineSemaphoreSubmitInfo timelineInfo{
			.waitSemaphoreValueCount = 1,
			.pWaitSemaphoreValues = &graphicsWaitValue,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = &graphicsSignalValue
		};

		vk::SubmitInfo submitInfo{
			.pNext = &timelineInfo,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*semaphore_,
			.pWaitDstStageMask = &graphicsWaitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &*cmds_.graphics[current_frame_],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*semaphore_
		};
		context_.queue_.submit(submitInfo, nullptr);
	}

	{
		vk::SemaphoreWaitInfo waitInfo{
			.semaphoreCount = 1,
			.pSemaphores = &*semaphore_,
			.pValues = &graphicsSignalValue
		};
		while (vk::Result::eTimeout == context_.device_.waitSemaphores(waitInfo, UINT64_MAX));
		vk::PresentInfoKHR presentInfo{
				.waitSemaphoreCount = 0, // No binary semaphores needed
				.pWaitSemaphores = nullptr,
				.swapchainCount = 1,
				.pSwapchains = &*swapchain_.swapchain_,
				.pImageIndices = &imageIndex
		};
		try {
			result = context_.queue_.presentKHR(presentInfo);
			if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || context_.framebuffer_resized_) {
				context_.framebuffer_resized_ = false;
				RecreateSwapchain();
			}
			else if (result != vk::Result::eSuccess) {
				throw std::runtime_error("failed to present swap chain image!");
			}
		}
		catch (const vk::SystemError& e) {
			if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
				RecreateSwapchain();
				return;
			}
			else {
				throw;
			}
		}
	}

	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void GraphicsContext::RecreateSwapchain()
{
	swapchain_.RecreateSwapChain(context_.physical_device_, context_.device_, context_.surface_);
	depth_image_ = nullptr;
	depth_image_memory_ = nullptr;
	depth_image_view_ = nullptr;
	CreateDepthResources();
}

void GraphicsContext::TransitionImageLayout(
	vk::Image& image,
	const vk::raii::CommandBuffer& cmd,
	vk::ImageLayout old_layout,
	vk::ImageLayout new_layout,
	vk::AccessFlags2 src_access_mask,
	vk::AccessFlags2 dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask
) {
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = src_stage_mask,
		.srcAccessMask = src_access_mask,
		.dstStageMask = dst_stage_mask,
		.dstAccessMask = dst_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};
	cmd.pipelineBarrier2(dependency_info);
}

void GraphicsContext::TransitionImageLayoutCustom(
	vk::raii::Image& image,
	const vk::raii::CommandBuffer& cmd,
	vk::ImageLayout old_layout,
	vk::ImageLayout new_layout,
	vk::AccessFlags2 src_access_mask,
	vk::AccessFlags2 dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask,
	vk::ImageAspectFlags aspect_mask
)
{
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = src_stage_mask,
		.srcAccessMask = src_access_mask,
		.dstStageMask = dst_stage_mask,
		.dstAccessMask = dst_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = *image,
		.subresourceRange = {
			.aspectMask = aspect_mask,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};
	cmd.pipelineBarrier2(dependency_info);
}

void GraphicsContext::UpdateGraphicsUBO(Camera& camera)
{
	// Global UBO 쓰기
	{
		const uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * graphics_.global_slot_size);
		auto* dst = static_cast<std::byte*>(graphics_.global_ubo_mapped) + globalOffset;

		graphics_.global_ubo_data.view = camera.View();
		graphics_.global_ubo_data.proj = camera.Proj(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height);

		std::memcpy(dst, &graphics_.global_ubo_data, sizeof(Graphics::GlobalUboData));
		// HostCoherent라 flush 생략, 비-coherent면 flush 필요
	}

	// Object UBO 쓰기
	{
		const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * graphics_.object_slot_size * model_manager_.kMaxObjects);
		for (uint32_t i = 0; i < model_manager_.model_count_; i++)
		{
			const uint32_t objOff = baseObjectOffset + i * static_cast<uint32_t>(graphics_.object_slot_size);
			auto* dst = static_cast<std::byte*>(graphics_.object_ubo_mapped) + objOff;

			graphics_.object_ubo_data.model = model_manager_.models[i]->world_;
			graphics_.object_ubo_data.color_use = model_manager_.models[i]->color_use_;

			std::memcpy(dst, &graphics_.object_ubo_data, sizeof(Graphics::ObjectUboData));
		}
	}
}

void GraphicsContext::RecordGraphicsCommandBuffer(uint32_t imageIndex)
{
	const auto& cmd = cmds_.graphics[current_frame_];

	cmd.reset();
	cmd.begin({});

	if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		cpu_sim_->CopyPositions(current_frame_, cmd);
	}

	TransitionImageLayout(
		swapchain_.swapchain_images_[imageIndex],
		cmd,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput
	);

	// Transition the depth image to DEPTH_ATTACHMENT_OPTIMAL
	TransitionImageLayoutCustom(
		depth_image_,
		cmd,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	vk::ClearValue clearColor = vk::ClearColorValue(background_color.r, background_color.g, background_color.b, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

	vk::RenderingAttachmentInfo colorAttachmentInfo = {
		.imageView = swapchain_.swapchain_image_views_[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};
	// Depth attachment
	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = depth_image_view_,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 },
		.extent = swapchain_.swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	cmd.beginRendering(renderingInfo);

	vk::Viewport vp(
		0.0f,
		0.0f,
		static_cast<float>(swapchain_.swapchain_extent_.width),
		static_cast<float>(swapchain_.swapchain_extent_.height),
		0.0f, 1.0f
	);
	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_.swapchain_extent_));

	uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * graphics_.global_slot_size);
	const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * graphics_.object_slot_size * model_manager_.kMaxObjects);

	if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		cpu_sim_->Record(current_frame_, cmd, graphics_.global_set, globalOffset);
	}
	else if (cpu_or_gpu_ == CpuOrGpu::GPU)
	{
		gpu_sim_->GraphicsRecord(current_frame_, cmd, graphics_.global_set, globalOffset);
	}

	// Model
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.model);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			graphics_.pipeline_layouts.model,
			0,
			{ *graphics_.global_set },
			{ globalOffset }
		);

		for (uint32_t i = 0; i < model_manager_.model_count_; ++i) {
			uint32_t objectOffset = baseObjectOffset + i * static_cast<uint32_t>(graphics_.object_slot_size);

			// Object set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				graphics_.pipeline_layouts.model,
				1,
				{ *graphics_.object_set },
				{ objectOffset }   // ← Set 1에도 동적 바인딩 1개 → 오프셋 1개만
			);

			cmd.bindVertexBuffers(0, { model_manager_.models[i]->mesh_data_.vertex_buffer }, { 0 });
			cmd.bindIndexBuffer(*model_manager_.models[i]->mesh_data_.index_buffer, 0, vk::IndexType::eUint32);
			cmd.drawIndexed(model_manager_.models[i]->mesh_data_.indices_count, 1, 0, 0, 0);
		}
	}

	// Imgui Render
	ImDrawData* draw_data = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(draw_data, *cmd);

	cmd.endRendering();

	// After rendering, transition the swapchain image to PRESENT_SRC
	TransitionImageLayout(
		swapchain_.swapchain_images_[imageIndex],
		cmd,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe
	);
	cmd.end();

}

vk::SampleCountFlagBits GraphicsContext::GetMaxUsableSampleCount() {
	vk::PhysicalDeviceProperties physicalDeviceProperties = context_.physical_device_.getProperties();

	vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
	if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
	if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
	if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
	if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
	if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

	return vk::SampleCountFlagBits::e1;
}

void GraphicsContext::CreateCommandBuffers()
{
	// Compute
	{
		cmds_.compute.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *context_.command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		cmds_.compute = vk::raii::CommandBuffers(context_.device_, allocInfo);
	}

	// Graphics
	{
		cmds_.graphics.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *context_.command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		cmds_.graphics = vk::raii::CommandBuffers(context_.device_, allocInfo);
	}
}

void GraphicsContext::CreateQueryPool() {
	vk::QueryPoolCreateInfo queryInfo = {};
	queryInfo.queryType = vk::QueryType::eTimestamp;
	queryInfo.queryCount = 4 + 10 * 40 + 2;

	timestamp_pool_ = context_.device_.createQueryPool(queryInfo);
}

void GraphicsContext::CreateDescriptorSetLayout()
{
	// Global UBO - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		graphics_.global_set_layout = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Object UBO + Sampler - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
		};
		counts_.ubo_dynamic += 1;
		counts_.sampler += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		graphics_.object_set_layout = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

}

void GraphicsContext::CreateDescriptorPools() {

	std::vector<vk::DescriptorPoolSize> poolSizes;

	if (counts_.ubo > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, counts_.ubo);
	}
	if (counts_.ubo_dynamic > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic, counts_.ubo_dynamic);
	}
	if (counts_.sampler > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, counts_.sampler);
	}
	if (counts_.sb > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, counts_.sb);
	}

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = counts_.layout,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	descriptor_pool_ = vk::raii::DescriptorPool(context_.device_, poolInfo);
}

void GraphicsContext::CreateUniformBuffers()
{
	// Global
	{
		graphics_.global_ubo.clear();
		graphics_.global_ubo_memory.clear();
		graphics_.global_ubo_mapped = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		graphics_.global_slot_size = (sizeof(Graphics::GlobalUboData) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = graphics_.global_slot_size * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		graphics_.global_ubo = std::move(buffer);
		graphics_.global_ubo_memory = std::move(bufferMem);
		graphics_.global_ubo_mapped = graphics_.global_ubo_memory.mapMemory(0, totalSize);
	}

	// Object
	{
		graphics_.object_ubo.clear();
		graphics_.object_ubo_memory.clear();
		graphics_.object_ubo_mapped = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		graphics_.object_slot_size = (sizeof(Graphics::ObjectUboData) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = graphics_.object_slot_size * MAX_FRAMES_IN_FLIGHT * model_manager_.kMaxObjects;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		graphics_.object_ubo = std::move(buffer);
		graphics_.object_ubo_memory = std::move(bufferMem);
		graphics_.object_ubo_mapped = graphics_.object_ubo_memory.mapMemory(0, totalSize);
	}

}

void GraphicsContext::CreateDescriptorSets()
{
	// Global UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*graphics_.global_set_layout
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		graphics_.global_set = std::move(sets.front());

		vk::DescriptorBufferInfo globalUboBufferInfo{ *graphics_.global_ubo, 0, sizeof(Graphics::GlobalUboData) };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *graphics_.global_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &globalUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Object UBO + Sampler
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*graphics_.object_set_layout
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		graphics_.object_set = std::move(sets.front());

		vk::DescriptorBufferInfo objectUboBufferInfo{ *graphics_.object_ubo, 0, sizeof(Graphics::ObjectUboData) };
		vk::DescriptorImageInfo imageInfo{
			.sampler = *texture_manager_.vulkan_title_image_->texture_sampler_,
			.imageView = *texture_manager_.vulkan_title_image_->texture_image_view_,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *graphics_.object_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &objectUboBufferInfo
			},
			vk::WriteDescriptorSet{
				.dstSet = *graphics_.object_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &imageInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}
}

void GraphicsContext::CreateGraphicsPipelines()
{
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		.topology = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False
	};
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1,
		.scissorCount = 1
	};
	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False
	};
	rasterizer.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = msaa_samples_,
		.sampleShadingEnable = vk::False
	};
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp = vk::CompareOp::eLess,
		.depthBoundsTestEnable = vk::False,
		.stencilTestEnable = vk::False
	};
	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	colorBlendAttachment.blendEnable = vk::False;

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	std::vector dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
	vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

	vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

	// Model
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/model.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/model.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"
		};
		vk::PipelineShaderStageCreateInfo fragStage{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = *fragModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages{ vertStage, fragStage };

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ false, false });

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vertexInputInfo.pVertexBindingDescriptions = vdesc.bindings.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vertexInputInfo.pVertexAttributeDescriptions = vdesc.attributes.data();

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 2> setLayouts(*graphics_.global_set_layout, *graphics_.object_set_layout);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 2, .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
		graphics_.pipeline_layouts.model = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		  {.stageCount = 2,
			.pStages = stages.data(),
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = graphics_.pipeline_layouts.model,
			.renderPass = nullptr },
		  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain_.swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
		};
		graphics_.pipelines.model = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

}

void GraphicsContext::CreateSyncObjects()
{
	in_flight_fences_.clear();

	vk::SemaphoreTypeCreateInfo semaphoreType{ .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0 };
	semaphore_ = vk::raii::Semaphore(context_.device_, { .pNext = &semaphoreType });
	timeline_value_ = 0;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::FenceCreateInfo fenceInfo{};
		in_flight_fences_.emplace_back(context_.device_, fenceInfo);
	}

}

void GraphicsContext::CreateDepthResources() {
	vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

	vku::CreateImage(context_.physical_device_, context_.device_, swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height, 1, msaa_samples_, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depth_image_, depth_image_memory_);
	depth_image_view_ = vku::CreateImageView(context_.device_, depth_image_, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}