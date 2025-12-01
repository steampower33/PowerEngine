#include "context.h"
#include "swapchain.h"
#include "cpu_sim.h"
#include "gpu_sim.h"
#include "camera.h"
#include "vertex.h"
#include "texture.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "gui.h"
#include "skybox.h"
#include "mouse_interactor.h"

#include "graphics_context.h"

GraphicsContext::GraphicsContext(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager)
	: context_(context), swapchain_(swapchain), texture_manager_(textureManager), model_manager_(modelManager)
{
	//msaa_samples_ = vku::GetMaxUsableSampleCount(context_.physical_device_.getProperties());

	CreateCommandBuffers();
	CreateQueryPool();

	CreateDescriptorSetLayout();
	CreateDescriptorPools();

	CreateUniformBuffers();

	CreateGeometryBuffers();
	CreateDepthResources();

	CreateDescriptorSets();
	CreateGraphicsPipelines();
	CreateSyncObjects();

	cpu_sim_ = std::make_unique<CpuSim>(context_, swapchain_, texture_manager_, model_manager_, set_layouts_.global, geometry_buffers_.formats, set_layouts_.tex2D);
	gpu_sim_ = std::make_unique<GpuSim>(context_, swapchain_, texture_manager_, model_manager_, set_layouts_.global, geometry_buffers_.formats, set_layouts_.tex2D);
}

void GraphicsContext::Update(Camera& camera, MouseInteractor& mouseInteractor)
{

	if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
	{
		cpu_sim_->UpdateGraphicsUBO(current_frame_);

		cpu_sim_->UpdateMousePushConstant(camera, mouseInteractor, glm::vec2(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height));

		cpu_sim_->ComputeSolve(
			model_manager_.models_[0]->position_, model_manager_.models_[0]->radius_
		);
	}
	else if (cpu_or_gpu_ == vku::CpuOrGpu::GPU)
	{
		gpu_sim_->UpdateComputeUBO(current_frame_, model_manager_.models_[0]);
		gpu_sim_->UpdateGraphicsUBO(current_frame_);

		gpu_sim_->UpdateMousePushConstant(camera, mouseInteractor, glm::vec2(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height));
	}

	UpdateGraphicsUBO(camera);
}

void GraphicsContext::UpdateGraphicsUBO(Camera& camera)
{
	// Global UBO
	{
		const uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.global);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.global) + globalOffset;

		ubo_datas_.global.view = camera.View();
		ubo_datas_.global.proj = camera.Proj(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height);

		std::memcpy(dst, &ubo_datas_.global, sizeof(UBOData::Global));
	}

	// Object UBO
	{
		const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.object * model_manager_.kMaxObjects);
		for (uint32_t i = 0; i < model_manager_.models_.size(); i++)
		{
			const uint32_t objOff = baseObjectOffset + i * static_cast<uint32_t>(ubo_size_.object);
			auto* dst = static_cast<std::byte*>(ubo_mapped_.object) + objOff;

			ubo_datas_.object.model = model_manager_.models_[i]->world_;
			ubo_datas_.object.albedo_use = model_manager_.models_[i]->albedo_use_;
			ubo_datas_.object.albedo_idx = model_manager_.models_[i]->texture_idx_.albedo;
			ubo_datas_.object.metallic_idx = model_manager_.models_[i]->texture_idx_.metallic;
			ubo_datas_.object.normal_idx = model_manager_.models_[i]->texture_idx_.normal;
			ubo_datas_.object.roughness_idx = model_manager_.models_[i]->texture_idx_.roughness;
			ubo_datas_.object.ao_idx = model_manager_.models_[i]->texture_idx_.ao;
			ubo_datas_.object.height_idx = model_manager_.models_[i]->texture_idx_.height;

			ubo_datas_.object.albedo_enable = model_manager_.models_[i]->texture_use_.albedo;
			ubo_datas_.object.metallic_enable = model_manager_.models_[i]->texture_use_.metallic;
			ubo_datas_.object.normal_enable = model_manager_.models_[i]->texture_use_.normal;
			ubo_datas_.object.roughness_enable = model_manager_.models_[i]->texture_use_.roughtness;
			ubo_datas_.object.ao_enable = model_manager_.models_[i]->texture_use_.ao;
			ubo_datas_.object.height_enable = model_manager_.models_[i]->texture_use_.height;

			ubo_datas_.object.checker_board_enable = model_manager_.models_[i]->checker_board_enable_;

			std::memcpy(dst, &ubo_datas_.object, sizeof(UBOData::Object));
		}
	}

	// Light
	{
		ubo_datas_.light.cameraPos = glm::vec4(camera.position, 0.0f);
		ubo_datas_.light.invViewProj = glm::inverse(ubo_datas_.global.proj * ubo_datas_.global.view);

		const uint32_t lightOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.light);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.light) + lightOffset;
		std::memcpy(dst, &ubo_datas_.light, sizeof(UBOData::Light));
	}
}

void GraphicsContext::RecordGraphicsCommandBuffer(uint32_t imageIndex, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps)
{
	const auto& cmd = cmds_.graphics[current_frame_];

	cmd.reset();
	cmd.begin({});

	timestampSteps = 0;
	uint32_t slots = 2;
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *timestampPool, idx++);
		};

	cmd.resetQueryPool(*timestampPool, 0, slots);

	TS(timestampSteps); // Start

	if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
	{
		cpu_sim_->CopyPositions(current_frame_, cmd);
	}

	auto toShaderWrite = [&](vk::raii::Image& img) {
		vku::TransitionImageLayoutCustom(
			img,
			cmd,
			first_frame_ ? vk::ImageLayout::eUndefined
			: vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eTopOfPipe,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::ImageAspectFlagBits::eColor
		);
		};

	// G-buffer: Undefined/ShaderReadOnly → ColorAttachmentOptimal
	toShaderWrite(geometry_buffers_.albedo_mettalic_image);
	toShaderWrite(geometry_buffers_.normal_roughness_image);
	toShaderWrite(geometry_buffers_.height_ao_image);

	vku::TransitionImageLayoutCustom(
		depth_image_,
		cmd,
		first_frame_ ? vk::ImageLayout::eUndefined
		: vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	std::array<vk::RenderingAttachmentInfo, 3> gbufferAttachments;

	auto renderingAttachmentInfo = [&](vk::raii::ImageView& imageView, vk::ClearValue clearColor) {
		return vk::RenderingAttachmentInfo{
			.imageView = *imageView,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.resolveMode = {},
			.resolveImageView = {},
			.resolveImageLayout = {},
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
		};
		};
	vk::ClearValue clearColor0 = vk::ClearColorValue(background_color_.r, background_color_.g, background_color_.b, 0.0f); // albedo+metal
	gbufferAttachments[0] = renderingAttachmentInfo(geometry_buffers_.albedo_mettalic_image_view, clearColor0);
	vk::ClearValue clearColor1 = vk::ClearColorValue(0.5f, 0.5f, 1.0f, 1.0f); // normal default (0,0,1)
	gbufferAttachments[1] = renderingAttachmentInfo(geometry_buffers_.normal_roughness_image_view, clearColor1);
	vk::ClearValue clearColor2 = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f); // height+AO
	gbufferAttachments[2] = renderingAttachmentInfo(geometry_buffers_.height_ao_image_view, clearColor2);

	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
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
		.colorAttachmentCount = static_cast<uint32_t>(gbufferAttachments.size()),
		.pColorAttachments = gbufferAttachments.data(),
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

	uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.global);
	const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.object * model_manager_.kMaxObjects);

	if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
	{
		cpu_sim_->RecordGraphics(current_frame_, cmd, sets_.global, globalOffset, polygon_mode_, sets_.tex2D);
	}
	else if (cpu_or_gpu_ == vku::CpuOrGpu::GPU)
	{
		gpu_sim_->RecordGraphics(current_frame_, cmd, sets_.global, globalOffset, polygon_mode_, sets_.tex2D);
	}

	// Model
	{
		if (polygon_mode_ == vku::PolygonMode::SOLID)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_solid);
		else if (polygon_mode_ == vku::PolygonMode::WIREFRAME)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_wireframe);
		else if (polygon_mode_ == vku::PolygonMode::POINT)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_point);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.model,
			0,
			{ *sets_.global },
			{ globalOffset }
		);

		// tex2D
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.model,
			2,
			{ *sets_.tex2D },
			{ }
		);

		for (uint32_t i = 0; i < model_manager_.models_.size(); ++i) {
			uint32_t objectOffset = baseObjectOffset + i * static_cast<uint32_t>(ubo_size_.object);

			// Object set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline_layouts_.model,
				1,
				{ *sets_.object },
				{ objectOffset }
			);

			cmd.bindVertexBuffers(0, { model_manager_.models_[i]->mesh_data_.vertex_buffer }, { 0 });
			cmd.bindIndexBuffer(*model_manager_.models_[i]->mesh_data_.index_buffer, 0, vk::IndexType::eUint32);
			cmd.drawIndexed(model_manager_.models_[i]->mesh_data_.indices_count, 1, 0, 0, 0);
		}
	}

	cmd.endRendering();

	auto toShaderRead = [&](vk::raii::Image& img) {
		vku::TransitionImageLayoutCustom(
			img,
			cmd,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::AccessFlagBits2::eShaderRead,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eColor
		);
		};
	toShaderRead(geometry_buffers_.albedo_mettalic_image);
	toShaderRead(geometry_buffers_.normal_roughness_image);
	toShaderRead(geometry_buffers_.height_ao_image);

	vku::TransitionImageLayoutCustom(
		depth_image_,
		cmd,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eShaderRead,
		vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth
	);

	vku::TransitionImageLayout(
		swapchain_.swapchain_images_[imageIndex],
		cmd,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput
	);

	// Lighting pass
	vk::ClearValue clearColor = vk::ClearColorValue(
		0.0f, 0.0f, 0.0f, 1.0f);

	vk::RenderingAttachmentInfo colorAttachmentInfo{
		.imageView = swapchain_.swapchain_image_views_[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.resolveMode = {},
		.resolveImageView = {},
		.resolveImageLayout = {},
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	vk::RenderingInfo lightingRenderingInfo{
		.renderArea = { {0, 0}, swapchain_.swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = nullptr
	};

	cmd.beginRendering(lightingRenderingInfo);

	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_.swapchain_extent_));

	// --- lighting pipeline bind ---
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.lighting);

	uint32_t lightOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.light);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		0,
		{ *sets_.lighting },
		{ lightOffset }
	);

	uint32_t skyboxOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.skybox);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		1,
		{ *sets_.skybox },
		{ skyboxOffset }
	);

	// tex2D
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		2,
		{ *sets_.tex2D },
		{ }
	);

	// texEnv
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		3,
		{ *sets_.texEnv },
		{ }
	);

	// fullscreen triangle
	cmd.draw(3, 1, 0, 0);

	//// Skybox
	//{
	//	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.skybox);

	//	// Global Set
	//	cmd.bindDescriptorSets(
	//		vk::PipelineBindPoint::eGraphics,
	//		pipeline_layouts_.skybox,
	//		0,
	//		{ *sets_.global },
	//		{ globalOffset }
	//	);

	//	// Skybox
	//	uint32_t skyboxOffset = static_cast<uint32_t>(current_frame_ * ubo_size_.skybox);
	//	cmd.bindDescriptorSets(
	//		vk::PipelineBindPoint::eGraphics,
	//		pipeline_layouts_.skybox,
	//		1,
	//		{ *sets_.skybox },
	//		{ skyboxOffset }
	//	);

	//	// texEnv
	//	cmd.bindDescriptorSets(
	//		vk::PipelineBindPoint::eGraphics,
	//		pipeline_layouts_.skybox,
	//		2,
	//		{ *sets_.texEnv },
	//		{ }
	//	);
	//	cmd.bindVertexBuffers(0, { model_manager_.skybox_->mesh_data_.vertex_buffer }, { 0 });
	//	cmd.bindIndexBuffer(*model_manager_.skybox_->mesh_data_.index_buffer, 0, vk::IndexType::eUint32);
	//	cmd.drawIndexed(model_manager_.skybox_->mesh_data_.indices_count, 1, 0, 0, 0);
	//}

	ImDrawData* draw_data = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(draw_data, *cmd);

	cmd.endRendering();

	// After rendering, transition the swapchain image to PRESENT_SRC
	vku::TransitionImageLayout(
		swapchain_.swapchain_images_[imageIndex],
		cmd,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe
	);


	TS(timestampSteps); // End

	cmd.end();

	frame_counter_++;
}

void GraphicsContext::Draw(std::unique_ptr<GUI>& gui)
{
	auto& device = context_.device_;
	auto& queue = context_.queue_;

	const uint32_t frame = current_frame_;

	{
		device.waitForFences(*in_flight_fences_[frame], vk::True, UINT64_MAX);
		device.resetFences(*in_flight_fences_[frame]);

		if (gui->is_print_timestamps)
		{
			vk::SemaphoreWaitInfo waitInfo{
				.semaphoreCount = 1,
				.pSemaphores = &*timeline_semaphore_,
				.pValues = &last_compute_timeline_
			};
			device.waitSemaphores(waitInfo, UINT64_MAX);

			float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
			float toMs = nsPerTick / 1e6f;

			uint32_t numTimestamp = compute_ts_steps_;
			std::vector<uint64_t> ts(numTimestamp);

			VkResult res = vkGetQueryPoolResults(
				static_cast<VkDevice>(*context_.device_),
				static_cast<VkQueryPool>(*compute_ts_pool_),
				0, numTimestamp,
				ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
				VK_QUERY_RESULT_64_BIT
			);

			auto delta_ms = [&](uint32_t i0, uint32_t i1) {
				return (ts[i1] - ts[i0]) * toMs;
				};

			float tIntegrate = 0.0f;
			float tClearLambdas = 0.0f;
			float tHashBuild = 0.0f;
			float tRadixSort = 0.0f;
			float tBuildCell = 0.0f;
			float tBuildNeighbor = 0.0f;
			float tSolveStretch = 0.0f;
			float tSolveShear = 0.0f;
			float tSolveBend = 0.0f;
			float tSolveArea = 0.0f;
			float tSolveSelfCollision = 0.0f;
			float tCollideSdf = 0.0f;
			float tApplyDeltas = 0.0f;
			float tUpdate = 0.0f;

			uint32_t tsCnt = gpu_sim_->iteration_timestamp_count_;
			uint32_t base = 1;
			for (uint32_t sub = 0; sub < gpu_sim_->datas_.substeps; sub++)
			{
				tIntegrate += delta_ms(base + 0, base + 1);
				tClearLambdas += delta_ms(base + 2, base + 3);
				tHashBuild += delta_ms(base + 4, base + 5);
				tRadixSort += delta_ms(base + 6, base + 7);
				tBuildCell += delta_ms(base + 8, base + 9);
				tBuildNeighbor += delta_ms(base + 10, base + 11);

				uint32_t iterBase = base + 12;
				for (uint32_t it = 0; it < gpu_sim_->datas_.iterations; it++)
				{
					tSolveStretch += delta_ms(iterBase + it * tsCnt + 0, iterBase + it * tsCnt + 1);
					tSolveShear += delta_ms(iterBase + it * tsCnt + 2, iterBase + it * tsCnt + 3);
					tSolveBend += delta_ms(iterBase + it * tsCnt + 4, iterBase + it * tsCnt + 5);
					tSolveArea += delta_ms(iterBase + it * tsCnt + 6, iterBase + it * tsCnt + 7);
					tSolveSelfCollision += delta_ms(iterBase + it * tsCnt + 8, iterBase + it * tsCnt + 9);
					tCollideSdf += delta_ms(iterBase + it * tsCnt + 10, iterBase + it * tsCnt + 11);
					tApplyDeltas += delta_ms(iterBase + it * tsCnt + 12, iterBase + it * tsCnt + 13);
				}
				uint32_t updateStart = iterBase + gpu_sim_->datas_.iterations * tsCnt;
				tUpdate += delta_ms(updateStart, updateStart + 1);
			}

			compute_all_time_ = delta_ms(0, numTimestamp - 1);

			//std::cout << compute_all_time_ << std::endl;

			float total =
				tIntegrate + tClearLambdas +
				tHashBuild + tRadixSort + tBuildCell + tBuildNeighbor +
				tSolveStretch + tSolveBend + tSolveArea + tSolveSelfCollision + tCollideSdf + tApplyDeltas +
				tUpdate;

			uint32_t c = 0;

			{
				c = 0;
				label_time_[labels_[c++]] = tIntegrate;
				label_time_[labels_[c++]] = tClearLambdas;
				label_time_[labels_[c++]] = tHashBuild;
				label_time_[labels_[c++]] = tRadixSort;
				label_time_[labels_[c++]] = tBuildCell;
				label_time_[labels_[c++]] = tBuildNeighbor;
				label_time_[labels_[c++]] = tSolveStretch;
				label_time_[labels_[c++]] = tSolveShear;
				label_time_[labels_[c++]] = tSolveBend;
				label_time_[labels_[c++]] = tSolveArea;
				label_time_[labels_[c++]] = tSolveSelfCollision;
				label_time_[labels_[c++]] = tCollideSdf;
				label_time_[labels_[c++]] = tApplyDeltas;
				label_time_[labels_[c++]] = tUpdate;
				label_time_[labels_[c++]] = total;
			}

			{
				c = 0;
				label_avg_time_[labels_[c++]] += tIntegrate;
				label_avg_time_[labels_[c++]] += tClearLambdas;
				label_avg_time_[labels_[c++]] += tHashBuild;
				label_avg_time_[labels_[c++]] += tRadixSort;
				label_avg_time_[labels_[c++]] += tBuildCell;
				label_avg_time_[labels_[c++]] += tBuildNeighbor;
				label_avg_time_[labels_[c++]] += tSolveStretch;
				label_avg_time_[labels_[c++]] += tSolveShear;
				label_avg_time_[labels_[c++]] += tSolveBend;
				label_avg_time_[labels_[c++]] += tSolveArea;
				label_avg_time_[labels_[c++]] += tSolveSelfCollision;
				label_avg_time_[labels_[c++]] += tCollideSdf;
				label_avg_time_[labels_[c++]] += tApplyDeltas;
				label_avg_time_[labels_[c++]] += tUpdate;
				label_avg_time_[labels_[c++]] += total;
			}

			time_count_++;

		}
	}

	uint32_t imageIndex = 0;
	{
		auto [result, idx] =
			swapchain_.swapchain_.acquireNextImage(
				UINT64_MAX,
				*image_available_[frame],
				nullptr
			);

		if (result == vk::Result::eErrorOutOfDateKHR) {
			RecreateSwapchain();
			return;
		}
		else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		imageIndex = idx;
	}

	// Record
	{
		gpu_sim_->RecordCompute(
			frame,
			cmds_.compute[frame],
			compute_ts_pool_,
			compute_ts_steps_,
			test_scene_);

		RecordGraphicsCommandBuffer(
			imageIndex,
			graphics_ts_pool_,
			graphics_ts_steps_);
	}

	// compute submit
	uint64_t computeSignalValue = 0;
	{
		computeSignalValue = ++timeline_value_;
		last_compute_timeline_ = computeSignalValue;

		vk::TimelineSemaphoreSubmitInfo timelineInfo{
			.waitSemaphoreValueCount = 0,
			.pWaitSemaphoreValues = nullptr,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = &computeSignalValue
		};

		vk::SubmitInfo submitInfo{
			.pNext = &timelineInfo,
			.waitSemaphoreCount = 0,
			.pWaitSemaphores = nullptr,
			.pWaitDstStageMask = nullptr,
			.commandBufferCount = 1,
			.pCommandBuffers = &*cmds_.compute[frame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*timeline_semaphore_
		};

		queue.submit(submitInfo, nullptr);
	}

	// graphics submit
	{
		vk::Semaphore waitSems[] = {
			*image_available_[frame]
		};
		vk::PipelineStageFlags waitStages[] = {
			vk::PipelineStageFlagBits::eColorAttachmentOutput
		};
		vk::Semaphore signalSems[] = {
			*image_render_finished_[imageIndex]
		};

		vk::SubmitInfo submitInfo{
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = waitSems,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &*cmds_.graphics[frame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signalSems
		};

		queue.submit(submitInfo, *in_flight_fences_[frame]);
	}

	{
		vk::Semaphore waitSem = *image_render_finished_[imageIndex];

		vk::PresentInfoKHR presentInfo{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &waitSem,
			.swapchainCount = 1,
			.pSwapchains = &*swapchain_.swapchain_,
			.pImageIndices = &imageIndex
		};

		auto result = queue.presentKHR(presentInfo);
		if (result == vk::Result::eErrorOutOfDateKHR ||
			result == vk::Result::eSuboptimalKHR ||
			context_.framebuffer_resized_) {
			context_.framebuffer_resized_ = false;
			RecreateSwapchain();
		}
		else if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to present swap chain image!");
		}
	}

	if (gui->is_print_timestamps)
	{
		float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
		float toMs = nsPerTick / 1e6f;

		uint32_t numTimestamp = graphics_ts_steps_;
		std::vector<uint64_t> ts(numTimestamp);

		VkResult res = vkGetQueryPoolResults(
			static_cast<VkDevice>(*context_.device_),
			static_cast<VkQueryPool>(*graphics_ts_pool_),
			0, numTimestamp,
			ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT
		);

		auto delta_ms = [&](uint32_t i0, uint32_t i1) {
			return (ts[i1] - ts[i0]) * toMs;
			};

		graphics_all_time_ = delta_ms(0, 1);
	}

	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
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
	queryInfo.queryCount = 2048;

	compute_ts_pool_ = context_.device_.createQueryPool(queryInfo);

	queryInfo.queryCount = 2;

	graphics_ts_pool_ = context_.device_.createQueryPool(queryInfo);
}

void GraphicsContext::CreateDescriptorSetLayout()
{
	// Global UBO - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.global = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Tex2D
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eCombinedImageSampler,
				texture_manager_.max_texture_size,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.sampler += texture_manager_.max_texture_size;
		counts_.layout += 1;

		std::array<vk::DescriptorBindingFlags, 1> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound |
			vk::DescriptorBindingFlagBits::eVariableDescriptorCount
		};

		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
			.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
			.pBindingFlags = bindingFlags.data()
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.pNext = &flagsInfo,
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.tex2D = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// TexEnv
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eCombinedImageSampler,
				texture_manager_.max_texture_size,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.sampler += texture_manager_.env_texture_size;
		counts_.layout += 1;

		std::array<vk::DescriptorBindingFlags, 1> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound |
			vk::DescriptorBindingFlagBits::eVariableDescriptorCount
		};

		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
			.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
			.pBindingFlags = bindingFlags.data()
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.pNext = &flagsInfo,
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.texEnv = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Object UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.object = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Light UBO + G-buffers
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				1,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				2,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				3,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				4,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};

		counts_.ubo += 1;
		counts_.sampler += 4;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.lighting =
			vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Skybox UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.skybox = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
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
		ubos_.global.clear();
		ubo_memories_.global.clear();
		ubo_mapped_.global = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.global = (sizeof(UBOData::Global) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.global * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.global = std::move(buffer);
		ubo_memories_.global = std::move(bufferMem);
		ubo_mapped_.global = ubo_memories_.global.mapMemory(0, totalSize);

		ubo_datas_.global.vulkan_thumbnail_index = texture_manager_.vulkan_thumbnail_index_;
	}

	// Object
	{
		ubos_.object.clear();
		ubo_memories_.object.clear();
		ubo_mapped_.object = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.object = (sizeof(UBOData::Object) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.object * MAX_FRAMES_IN_FLIGHT * model_manager_.kMaxObjects;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.object = std::move(buffer);
		ubo_memories_.object = std::move(bufferMem);
		ubo_mapped_.object = ubo_memories_.object.mapMemory(0, totalSize);
	}

	// Light
	{
		ubos_.light.clear();
		ubo_memories_.light.clear();
		ubo_mapped_.light = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.light = (sizeof(UBOData::Light) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.light * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.light = std::move(buffer);
		ubo_memories_.light = std::move(bufferMem);
		ubo_mapped_.light = ubo_memories_.light.mapMemory(0, totalSize);
	}

	// Skybox
	{
		ubos_.skybox.clear();
		ubo_memories_.skybox.clear();
		ubo_mapped_.skybox = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.skybox = (sizeof(UBOData::SkyBox) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.skybox * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.skybox = std::move(buffer);
		ubo_memories_.skybox = std::move(bufferMem);
		ubo_mapped_.skybox = ubo_memories_.skybox.mapMemory(0, totalSize);

		ubo_datas_.skybox.model = model_manager_.skybox_->world_;
		ubo_datas_.skybox.envIdx = model_manager_.skybox_->texture_idx_.env;
		ubo_datas_.skybox.radianceIdx = model_manager_.skybox_->texture_idx_.radiance;
		ubo_datas_.skybox.irradianceIdx = model_manager_.skybox_->texture_idx_.irradiance;
		ubo_datas_.skybox.brdfLUTIndex = texture_manager_.brdf_lut_index_;
		ubo_datas_.skybox.specularMipLevels = texture_manager_.textures_[model_manager_.skybox_->texture_idx_.radiance]->mip_levels_;

		auto* dst = static_cast<std::byte*>(ubo_mapped_.skybox);
		std::memcpy(dst, &ubo_datas_.skybox, ubo_size_.skybox);

		dst += ubo_size_.skybox;
		std::memcpy(dst, &ubo_datas_.skybox, ubo_size_.skybox);
	}
}

void GraphicsContext::CreateDescriptorSets()
{
	// Global UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.global
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.global = std::move(sets.front());

		vk::DescriptorBufferInfo globalUboBufferInfo{ *ubos_.global, 0, sizeof(UBOData::Global) };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *sets_.global,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &globalUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Tex2D
	{
		// Create
		uint32_t maxTextures = texture_manager_.max_texture_size;

		vk::DescriptorSetVariableDescriptorCountAllocateInfo countInfo{
			.descriptorSetCount = 1,
			.pDescriptorCounts = &maxTextures
		};

		vk::DescriptorSetAllocateInfo allocInfo{
			.pNext = &countInfo,
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.tex2D
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.tex2D = std::move(sets.front());

		// Update
		std::vector<vk::DescriptorImageInfo> imageInfos;
		for (auto& tex : texture_manager_.textures_) {
			imageInfos.push_back(vk::DescriptorImageInfo{
				.sampler = *tex->texture_sampler_,
				.imageView = *tex->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
				});
		}
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.tex2D,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<uint32_t>(imageInfos.size()),
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = imageInfos.data(),
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});

	}

	// TexEnv
	{
		// Create
		uint32_t maxTextures = texture_manager_.env_texture_size;

		vk::DescriptorSetVariableDescriptorCountAllocateInfo countInfo{
			.descriptorSetCount = 1,
			.pDescriptorCounts = &maxTextures
		};

		vk::DescriptorSetAllocateInfo allocInfo{
			.pNext = &countInfo,
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.texEnv
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.texEnv = std::move(sets.front());

		// Update
		std::vector<vk::DescriptorImageInfo> imageInfos;
		for (auto& tex : texture_manager_.env_textures_) {
			imageInfos.push_back(vk::DescriptorImageInfo{
				.sampler = *tex->texture_sampler_,
				.imageView = *tex->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
				});
		}
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.texEnv,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<uint32_t>(imageInfos.size()),
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = imageInfos.data(),
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Object UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.object
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.object = std::move(sets.front());

		// Update
		vk::DescriptorBufferInfo objectUboBufferInfo{ *ubos_.object, 0, sizeof(UBOData::Object) };

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.object,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &objectUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Lighting G-buffers
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.lighting
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.lighting = std::move(sets.front());

		vk::DescriptorBufferInfo lightUboBufferInfo{ *ubos_.light, 0, sizeof(UBOData::Light) };

		std::array<vk::DescriptorImageInfo, 4> gbufferInfos{
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.albedo_mettalic_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.normal_roughness_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.height_ao_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
		};

		vk::DescriptorImageInfo depthImageInfo{
			.sampler = *geometry_buffers_.sampler,
			.imageView = *depth_image_view_,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &lightUboBufferInfo
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[0]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[1]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[2]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &depthImageInfo
			}
		};

		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Skybox
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.skybox
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.skybox = std::move(sets.front());

		vk::DescriptorBufferInfo skyboxUboBufferInfo{ *ubos_.skybox, 0, sizeof(UBOData::SkyBox) };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *sets_.skybox,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &skyboxUboBufferInfo
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
	vk::PipelineRasterizationStateCreateInfo rasterizerSolid{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False,
		.lineWidth = 1.0f
	};
	vk::PipelineRasterizationStateCreateInfo rasterizerWireframe = rasterizerSolid;
	rasterizerWireframe.polygonMode = vk::PolygonMode::eLine;
	rasterizerWireframe.cullMode = vk::CullModeFlagBits::eNone;
	vk::PipelineRasterizationStateCreateInfo rasterizerPoint = rasterizerWireframe;
	rasterizerPoint.polygonMode = vk::PolygonMode::ePoint;

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

	std::array<vk::PipelineColorBlendAttachmentState, 3> colorBlendAttachments{};
	for (auto& a : colorBlendAttachments) {
		a.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;
		a.blendEnable = vk::False;
	}

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = colorBlendAttachments.size(),
		.pAttachments = colorBlendAttachments.data()
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
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ true, true, true });

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vertexInputInfo.pVertexBindingDescriptions = vdesc.bindings.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vertexInputInfo.pVertexAttributeDescriptions = vdesc.attributes.data();

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 3> setLayouts(*set_layouts_.global, *set_layouts_.object, *set_layouts_.tex2D);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = setLayouts.size(), .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
		pipeline_layouts_.model = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		// Pipeline
		{
			vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
			{
				{
					.stageCount = 2,
					.pStages = stages.data(),
					.pVertexInputState = &vertexInputInfo,
					.pInputAssemblyState = &inputAssembly,
					.pViewportState = &viewportState,
					.pRasterizationState = &rasterizerSolid,
					.pMultisampleState = &multisampling,
					.pDepthStencilState = &depthStencil,
					.pColorBlendState = &colorBlending,
					.pDynamicState = &dynamicState,
					.layout = pipeline_layouts_.model,
					.renderPass = nullptr
				},
				{
				  .colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
				  .pColorAttachmentFormats = geometry_buffers_.formats.data(),
				  .depthAttachmentFormat = depthFormat
				}
			};
			pipelines_.model_solid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}

		{
			vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
			{
				{
					.stageCount = 2,
					.pStages = stages.data(),
					.pVertexInputState = &vertexInputInfo,
					.pInputAssemblyState = &inputAssembly,
					.pViewportState = &viewportState,
					.pRasterizationState = &rasterizerWireframe,
					.pMultisampleState = &multisampling,
					.pDepthStencilState = &depthStencil,
					.pColorBlendState = &colorBlending,
					.pDynamicState = &dynamicState,
					.layout = pipeline_layouts_.model,
					.renderPass = nullptr
				},
				{
				  .colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
				  .pColorAttachmentFormats = geometry_buffers_.formats.data(),
				  .depthAttachmentFormat = depthFormat
				}
			};
			pipelines_.model_wireframe = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}

		{
			vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
			{
				{
					.stageCount = 2,
					.pStages = stages.data(),
					.pVertexInputState = &vertexInputInfo,
					.pInputAssemblyState = &inputAssembly,
					.pViewportState = &viewportState,
					.pRasterizationState = &rasterizerPoint,
					.pMultisampleState = &multisampling,
					.pDepthStencilState = &depthStencil,
					.pColorBlendState = &colorBlending,
					.pDynamicState = &dynamicState,
					.layout = pipeline_layouts_.model,
					.renderPass = nullptr
				},
				{
				  .colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
				  .pColorAttachmentFormats = geometry_buffers_.formats.data(),
				  .depthAttachmentFormat = depthFormat
				}
			};
			pipelines_.model_point = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}
	}

	// Lighting pipeline
	{
		auto vertCode = vku::ReadFile("shaders/spv/lighting.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/lighting.frag.spv");

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

		// fullscreen triangle: no vertex input
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

		// color blend: attachment 1
		vk::PipelineColorBlendAttachmentState lightingBlendAttachment{};
		lightingBlendAttachment.blendEnable = vk::False;
		lightingBlendAttachment.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo lightingColorBlending{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &lightingBlendAttachment
		};

		std::array<vk::DescriptorSetLayout, 4> setLayouts{
			*set_layouts_.lighting,
			*set_layouts_.skybox,
			*set_layouts_.tex2D,
			*set_layouts_.texEnv
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 0
		};
		pipeline_layouts_.lighting =
			vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::Format swapchainFormat = swapchain_.swapchain_surface_format_.format;

		rasterizerSolid.cullMode = vk::CullModeFlagBits::eNone;
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{.stageCount = 2,
			  .pStages = stages.data(),
			  .pVertexInputState = &vertexInputInfo,
			  .pInputAssemblyState = &inputAssembly,
			  .pViewportState = &viewportState,
			  .pRasterizationState = &rasterizerSolid,
			  .pMultisampleState = &multisampling,
			  .pDepthStencilState = nullptr,
			  .pColorBlendState = &lightingColorBlending,
			  .pDynamicState = &dynamicState,
			  .layout = pipeline_layouts_.lighting,
			  .renderPass = nullptr },
			{.colorAttachmentCount = 1,
			  .pColorAttachmentFormats = &swapchainFormat,
			  .depthAttachmentFormat = vk::Format::eUndefined }
		};

		pipelines_.lighting = vk::raii::Pipeline(
			context_.device_, nullptr,
			pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Skybox
	{
		auto vertCode = vku::ReadFile("shaders/spv/skybox.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/skybox.frag.spv");

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

		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ false, false, false });

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vertexInputInfo.pVertexBindingDescriptions = vdesc.bindings.data();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vertexInputInfo.pVertexAttributeDescriptions = vdesc.attributes.data();

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 3> setLayouts(
			*set_layouts_.global,
			*set_layouts_.skybox,
			*set_layouts_.texEnv);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = setLayouts.size(), .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
		pipeline_layouts_.skybox = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRasterizationStateCreateInfo skyboxRasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineDepthStencilStateCreateInfo skyboxDepthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::False,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		// color blend: attachment 1
		vk::PipelineColorBlendAttachmentState skyboxBlendAttachment{};
		skyboxBlendAttachment.blendEnable = vk::False;
		skyboxBlendAttachment.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo skyboxColorBlending{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &skyboxBlendAttachment
		};


		vk::Format swapchainFormat = swapchain_.swapchain_surface_format_.format;

		// Pipeline
		{
			vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
			{
				{
					.stageCount = 2,
					.pStages = stages.data(),
					.pVertexInputState = &vertexInputInfo,
					.pInputAssemblyState = &inputAssembly,
					.pViewportState = &viewportState,
					.pRasterizationState = &skyboxRasterizer,
					.pMultisampleState = &multisampling,
					.pDepthStencilState = &skyboxDepthStencil,
					.pColorBlendState = &skyboxColorBlending,
					.pDynamicState = &dynamicState,
					.layout = pipeline_layouts_.skybox,
					.renderPass = nullptr
				},
				{
				  .colorAttachmentCount = static_cast<uint32_t>(1),
				  .pColorAttachmentFormats = &swapchainFormat,
				  .depthAttachmentFormat = depthFormat
				}
			};
			pipelines_.skybox = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}
	}
}

void GraphicsContext::CreateSyncObjects()
{
	in_flight_fences_.clear();

	vk::SemaphoreTypeCreateInfo semaphoreType{ 
		.semaphoreType = vk::SemaphoreType::eTimeline, 
		.initialValue = 0 
	};
	timeline_semaphore_ = vk::raii::Semaphore(context_.device_, { .pNext = &semaphoreType });
	timeline_value_ = 0;

	image_available_.clear();
	in_flight_fences_.clear();

	image_available_.reserve(MAX_FRAMES_IN_FLIGHT);
	in_flight_fences_.reserve(MAX_FRAMES_IN_FLIGHT);

	vk::SemaphoreCreateInfo binarySemaphoreInfo{}; // 기본은 binary
	vk::FenceCreateInfo fenceInfo{
		.flags = vk::FenceCreateFlagBits::eSignaled // 첫 프레임용
	};

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		image_available_.emplace_back(context_.device_, binarySemaphoreInfo);
		in_flight_fences_.emplace_back(context_.device_, fenceInfo);

		frame_timeline_done_[i] = 0;  // 해당 프레임이 마지막으로 기다린 타임라인 값
	}

	image_render_finished_.clear();
	image_render_finished_.reserve(swapchain_.image_count_);
	for (size_t i = 0; i < swapchain_.image_count_; i++)
	{

		image_render_finished_.emplace_back(context_.device_, binarySemaphoreInfo);
	}
}

void GraphicsContext::CreateGeometryBuffers()
{
	//RT0: albedo + metallic → VK_FORMAT_R8G8B8A8_UNORM
	{
		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		auto& image = geometry_buffers_.albedo_mettalic_image;
		auto& imageView = geometry_buffers_.albedo_mettalic_image_view;
		auto& memory = geometry_buffers_.albedo_mettalic_image_memory;
		geometry_buffers_.formats.push_back(format);
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	//RT1 : normal + roughness → VK_FORMAT_A2B10G10R10_UNORM_PACK32 or VK_FORMAT_R16G16B16A16_SFLOAT
	{
		vk::Format format = vk::Format::eR16G16B16A16Sfloat;
		geometry_buffers_.formats.push_back(format);
		auto& image = geometry_buffers_.normal_roughness_image;
		auto& imageView = geometry_buffers_.normal_roughness_image_view;
		auto& memory = geometry_buffers_.normal_roughness_image_memory;
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	//RT2 : height + ao → VK_FORMAT_R16G16_SFLOAT or VK_FORMAT_R8G8_UNORM
	{
		vk::Format format = vk::Format::eR8G8Unorm;
		geometry_buffers_.formats.push_back(format);
		auto& image = geometry_buffers_.height_ao_image;
		auto& imageView = geometry_buffers_.height_ao_image_view;
		auto& memory = geometry_buffers_.height_ao_image_memory;
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eNearest,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.minLod = 0.0f,
		.maxLod = 0.0f
	};
	geometry_buffers_.sampler = vk::raii::Sampler(context_.device_, samplerInfo);
}

void GraphicsContext::CreateDepthResources() {
	vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

	vku::CreateImage(context_.physical_device_, context_.device_, swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height, 1, msaa_samples_, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, depth_image_, depth_image_memory_);
	depth_image_view_ = vku::CreateImageView(context_.device_, depth_image_, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void GraphicsContext::RecreateSwapchain()
{
	swapchain_.RecreateSwapChain(context_.physical_device_, context_.device_, context_.surface_);
	depth_image_ = nullptr;
	depth_image_memory_ = nullptr;
	depth_image_view_ = nullptr;
	CreateDepthResources();
}


GraphicsContext::~GraphicsContext()
{

}
