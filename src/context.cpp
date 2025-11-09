#include "swapchain.h"
#include "vulkan_utils.h"
#include "vertex.h"
#include "camera.h"
#include "model.h"
#include "texture_2d.h"
#include "mouse_interactor.h"
#include "cpu_sim.h"
#include "gpu_sim.h"

#include "context.h"

Context::Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height)
	: glfw_window_(glfwWindow)
{
	CreateInstance();
	SetupDebugMessenger();
	CreateSurface();
	PickPhysicalDevice();
	//msaa_samples_ = GetMaxUsableSampleCount();
	CreateLogicalDevice();
	swapchain_ = std::make_unique<Swapchain>(glfw_window_, device_, physical_device_, msaa_samples_, surface_);
	CreateCommandPool();
	CreateCommandBuffers();
	CreateQueryPool();

	{
		models.reserve(kMaxObjects);

		models.emplace_back(std::make_unique<Model>("assets/models/sphere.gltf", physical_device_, device_, queue_, command_pool_, model_count_, glm::vec3(0.0f, 0.0f, 0.0f)));

		texture_ = std::make_unique<Texture2D>("assets/textures/vulkan_cloth_rgba.ktx", physical_device_, device_, queue_, command_pool_);
	}

	CreateDescriptorSetLayout();
	CreateDescriptorPools();

	CreateUniformBuffers();

	CreateDescriptorSets();
	CreateGraphicsPipelines();
	CreateSyncObjects();

	CreateDepthResources();

	SetupImgui(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height);

	cpu_sim_ = std::make_unique<CpuSim>(physical_device_, device_, queue_, command_pool_, swapchain_, Nx_, Ny_, spacing_, texture_, graphics_.global_set_layout);
	gpu_sim_ = std::make_unique<GpuSim>(physical_device_, device_, queue_, command_pool_, swapchain_, Nx_, Ny_, spacing_, texture_, graphics_.global_set_layout);
}

void Context::WaitIdle()
{
	device_.waitIdle();
}

Context::~Context()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Context::Update(Camera& camera, MouseInteractor& mouse_interactor, float dt)
{
	UpdateMouseInteractor(camera, mouse_interactor);
	
	if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		cpu_sim_->SimulateClothXPBD_CPU(
			models[0]->position_, models[0]->radius_
		);
	}
	else if (cpu_or_gpu_ == CpuOrGpu::GPU)
	{
		gpu_sim_->UpdateComputeUBO(current_frame_, models[0]);
	}

	UpdateGraphicsUBO(camera);
}

void Context::Draw(bool& printTimestamp)
{
	DrawImgui();

	auto [result, imageIndex] = swapchain_->swapchain_.acquireNextImage(UINT64_MAX, nullptr, in_flight_fences_[current_frame_]);

	while (vk::Result::eTimeout == device_.waitForFences(*in_flight_fences_[current_frame_], vk::True, UINT64_MAX));
	device_.resetFences(*in_flight_fences_[current_frame_]);

	uint64_t computeWaitValue = timeline_value_;
	uint64_t computeSignalValue = ++timeline_value_;
	uint64_t graphicsWaitValue = computeSignalValue;
	uint64_t graphicsSignalValue = ++timeline_value_;
	
	gpu_sim_->ComputeRecord(current_frame_, cmds_.compute[current_frame_], timestamp_pool_, steps);
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

		queue_.submit(computeSubmitInfo, nullptr);
	}

	if (printTimestamp)
	{
		printTimestamp = false;
		uint32_t prev = (current_frame_ + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;

		vk::SemaphoreWaitInfo waitInfo{
			.semaphoreCount = 1,
			.pSemaphores = &*semaphore_,
			.pValues = &computeSignalValue // 또는 computeSignalValue
		};
		while (vk::Result::eTimeout == device_.waitSemaphores(waitInfo, UINT64_MAX));

		std::array<uint64_t, 8> ts;

		VkResult res = vkGetQueryPoolResults(
			static_cast<VkDevice>(*device_),
			static_cast<VkQueryPool>(*timestamp_pool_),
			0, 8,
			sizeof(ts), ts.data(), sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT
		);
		float nsPerTick = physical_device_.getProperties().limits.timestampPeriod;

		std::cout << "===============================" << std::endl;
		for (int i = 0; i < 8; i += 2) {
			double dt_ms = (ts[i + 1] - ts[i]) * nsPerTick / 1e6;

			switch (i)
			{
				case 0:
					std::cout << "Clear Deltas \t: ";
					break;
				case 2:
					std::cout << "Solve XPBD \t: ";
					break;
				case 4:
					std::cout << "Apply Deltas \t: ";
					break;
				case 6:
					std::cout << "Collide Sphere \t: ";
					break;
			}

			std::cout << std::format("{:.3f} ms\n", dt_ms);
		}
	}

	//uint64_t graphicsWaitValue = timeline_value_;
	//uint64_t graphicsSignalValue = ++timeline_value_;

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
		queue_.submit(submitInfo, nullptr);
	}

	{
		vk::SemaphoreWaitInfo waitInfo{
			.semaphoreCount = 1,
			.pSemaphores = &*semaphore_,
			.pValues = &graphicsSignalValue
		};
		while (vk::Result::eTimeout == device_.waitSemaphores(waitInfo, UINT64_MAX));
		vk::PresentInfoKHR presentInfo{
				.waitSemaphoreCount = 0, // No binary semaphores needed
				.pWaitSemaphores = nullptr,
				.swapchainCount = 1,
				.pSwapchains = &*swapchain_->swapchain_,
				.pImageIndices = &imageIndex
		};
		try {
			result = queue_.presentKHR(presentInfo);
			if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebuffer_resized_) {
				framebuffer_resized_ = false;
				swapchain_->RecreateSwapChain(physical_device_, device_, surface_);
				depth_image_ = nullptr;
				depth_image_memory_ = nullptr;
				depth_image_view_ = nullptr;
				CreateDepthResources();
			}
			else if (result != vk::Result::eSuccess) {
				throw std::runtime_error("failed to present swap chain image!");
			}
		}
		catch (const vk::SystemError& e) {
			if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
				swapchain_->RecreateSwapChain(physical_device_, device_, surface_);
				depth_image_ = nullptr;
				depth_image_memory_ = nullptr;
				depth_image_view_ = nullptr;
				CreateDepthResources();
				return;
			}
			else {
				throw;
			}
		}
	}

	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Context::TransitionImageLayout(
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

void Context::TransitionImageLayoutCustom(
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

void Context::DrawImgui()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	{
		ImGui::Begin("Main");

		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		ImGui::SliderFloat("dt", &gpu_sim_->compute_.sim_params.dt, 0.001f, 0.008f, "%.4f");
		ImGui::SliderFloat("compliance", &gpu_sim_->compute_.sim_params.compliance, 0.0f, 5e-4f, "%.1e");
		ImGui::SliderFloat("damping", &gpu_sim_->compute_.sim_params.damping, 0.0f, 0.2f, "%.3f");
		ImGui::SliderFloat("collisionBeta", &gpu_sim_->compute_.sim_params.collisionBeta, 0.0f, 1.0f);

		ImGui::End();
	}

	ImGui::Render();
}

void Context::UpdateMouseInteractor(Camera& camera, MouseInteractor& mouse_interactor)
{
	mouse_interactor.Update(camera, glm::vec2(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height), models);
}

void Context::UpdateGraphicsUBO(Camera& camera)
{
	// Global UBO 쓰기
	{
		const uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * graphics_.global_slot_size);
		auto* dst = static_cast<std::byte*>(graphics_.global_ubo_mapped) + globalOffset;

		graphics_.global_ubo_data.view = camera.View();
		graphics_.global_ubo_data.proj = camera.Proj(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height);

		std::memcpy(dst, &graphics_.global_ubo_data, sizeof(Graphics::GlobalUboData));
		// HostCoherent라 flush 생략, 비-coherent면 flush 필요
	}

	// Object UBO 쓰기
	{
		const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * graphics_.object_slot_size * kMaxObjects);
		for (uint32_t i = 0; i < model_count_; i++)
		{
			const uint32_t objOff = baseObjectOffset + i * static_cast<uint32_t>(graphics_.object_slot_size);
			auto* dst = static_cast<std::byte*>(graphics_.object_ubo_mapped) + objOff;

			graphics_.object_ubo_data.model = models[i]->world_;

			std::memcpy(dst, &graphics_.object_ubo_data, sizeof(Graphics::ObjectUboData));
		}
	}
}

void Context::RecordGraphicsCommandBuffer(uint32_t imageIndex)
{
	const auto& cmd = cmds_.graphics[current_frame_];

	cmd.reset();
	cmd.begin({});

	if (cpu_or_gpu_ == CpuOrGpu::CPU)
	{
		cpu_sim_->CopyPositions(current_frame_, cmd);
	}

	TransitionImageLayout(
		swapchain_->swapchain_images_[imageIndex],
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

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

	vk::RenderingAttachmentInfo colorAttachmentInfo = {
		.imageView = swapchain_->swapchain_image_views_[imageIndex],
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
		.extent = swapchain_->swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	cmd.beginRendering(renderingInfo);

	vk::Viewport vp(
		0.0f,
		0.0f,
		static_cast<float>(swapchain_->swapchain_extent_.width),
		static_cast<float>(swapchain_->swapchain_extent_.height), // height = -H
		0.0f, 1.0f
	);
	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_->swapchain_extent_));

	uint32_t globalOffset = static_cast<uint32_t>(current_frame_ * graphics_.global_slot_size);
	const uint32_t baseObjectOffset = static_cast<uint32_t>(current_frame_ * graphics_.object_slot_size * kMaxObjects);

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

		for (uint32_t i = 0; i < model_count_; ++i) {
			uint32_t objectOffset = baseObjectOffset + i * static_cast<uint32_t>(graphics_.object_slot_size);

			std::array<uint32_t, 2> dynOffsets{ globalOffset, objectOffset };

			// Object set
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				graphics_.pipeline_layouts.model,
				1,
				{ *graphics_.object_set },
				{ objectOffset }   // ← Set 1에도 동적 바인딩 1개 → 오프셋 1개만
			);

			cmd.bindVertexBuffers(0, { models[i]->vertex_buffer_ }, { 0 });
			cmd.bindIndexBuffer(*models[i]->index_buffer_, 0, vk::IndexType::eUint32);
			cmd.drawIndexed(models[i]->indices_.size(), 1, 0, 0, 0);
		}
	}

	// Imgui Render
	ImDrawData* draw_data = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(draw_data, *cmd);

	cmd.endRendering();

	// After rendering, transition the swapchain image to PRESENT_SRC
	TransitionImageLayout(
		swapchain_->swapchain_images_[imageIndex],
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

void Context::CreateInstance() {
	constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Power Engine",
				.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
				.pEngineName = "Power Engine",
				.engineVersion = VK_MAKE_VERSION(1, 0, 0),
				.apiVersion = vk::ApiVersion14 };

	// Get the required layers
	std::vector<char const*> requiredLayers;
	if (enableValidationLayers) {
		requiredLayers.assign(validation_layers.begin(), validation_layers.end());
	}

	vk::raii::Context context;
	// Check if the required layers are supported by the Vulkan implementation.
	auto layerProperties = context.enumerateInstanceLayerProperties();
	for (auto const& requiredLayer : requiredLayers)
	{
		if (std::ranges::none_of(layerProperties,
			[requiredLayer](auto const& layerProperty)
			{ return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
		{
			throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
		}
	}

	// Get the required extensions.
	auto requiredExtensions = GetRequiredExtensions();

	// Check if the required extensions are supported by the Vulkan implementation.
	auto extensionProperties = context.enumerateInstanceExtensionProperties();
	for (auto const& requiredExtension : requiredExtensions)
	{
		if (std::ranges::none_of(extensionProperties,
			[requiredExtension](auto const& extensionProperty)
			{ return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
		{
			throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
		}
	}

	vk::InstanceCreateInfo createInfo{
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
		.ppEnabledLayerNames = requiredLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data() };
	instance_ = vk::raii::Instance(context, createInfo);
}

std::vector<const char*> Context::GetRequiredExtensions() {
	uint32_t glfwExtensionCount = 0;
	auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	if (enableValidationLayers) {
		extensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	return extensions;
}

void Context::SetupDebugMessenger() {
	if (!enableValidationLayers) return;

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags,
		.messageType = messageTypeFlags,
		.pfnUserCallback = &DebugCallback
	};
	debug_messenger_ = instance_.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
	if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
		std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
	}

	return vk::False;
}

void Context::CreateSurface() {
	VkSurfaceKHR       _surface;
	if (glfwCreateWindowSurface(*instance_, glfw_window_, nullptr, &_surface) != 0) {
		throw std::runtime_error("failed to create window surface!");
	}
	surface_ = vk::raii::SurfaceKHR(instance_, _surface);

}

void Context::PickPhysicalDevice() {
	std::vector<vk::raii::PhysicalDevice> devices = instance_.enumeratePhysicalDevices();
	const auto                            devIter = std::ranges::find_if(
		devices,
		[&](auto const& device)
		{
			// Check if the device supports the Vulkan 1.3 API version
			bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

			// Check if any of the queue families support graphics operations
			auto queueFamilies = device.getQueueFamilyProperties();
			bool supportsGraphics =
				std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

			// Check if all required device extensions are available
			auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
			auto hasAtomicFloatExt = std::any_of(
				availableDeviceExtensions.begin(), availableDeviceExtensions.end(),
				[](const vk::ExtensionProperties& e) {
					return std::strcmp(e.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0;
				});

			bool supportsAllRequiredExtensions =
				std::ranges::all_of(required_device_extension_,
					[&availableDeviceExtensions](auto const& requiredDeviceExtension)
					{
						return std::ranges::any_of(availableDeviceExtensions,
							[requiredDeviceExtension](auto const& availableDeviceExtension)
							{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
					});

			auto features = device.template getFeatures2<
				vk::PhysicalDeviceFeatures2,
				vk::PhysicalDeviceVulkan13Features,
				vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
				vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
				vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
			bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
				features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
				features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
				features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;

			auto atomicFeats = features.template get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
			bool bufF32Atomics = atomicFeats.shaderBufferFloat32Atomics;      // SSBO float32 원자연산
			bool bufF32AtomicAdd = atomicFeats.shaderBufferFloat32AtomicAdd;    // SSBO float32 atomicAdd
			bool sharedF32Atomics = atomicFeats.shaderSharedFloat32Atomics;      // shared memory float32 원자
			bool sharedF32AtomicAdd = atomicFeats.shaderSharedFloat32AtomicAdd;    // shared memory float32 add

			bool supportsRequiredAtomicFeatures = bufF32Atomics && bufF32AtomicAdd && sharedF32Atomics && sharedF32AtomicAdd;

			return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures && supportsRequiredAtomicFeatures;
		});
	if (devIter != devices.end())
	{
		physical_device_ = *devIter;
	}
	else
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}

vk::SampleCountFlagBits Context::GetMaxUsableSampleCount() {
	vk::PhysicalDeviceProperties physicalDeviceProperties = physical_device_.getProperties();

	vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
	if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
	if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
	if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
	if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
	if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

	return vk::SampleCountFlagBits::e1;
}


void Context::CreateLogicalDevice() {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physical_device_.getQueueFamilyProperties();

	// get the first index into queueFamilyProperties which supports both graphics and present
	for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
	{
		if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			(queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
			physical_device_.getSurfaceSupportKHR(qfpIndex, *surface_))
		{
			// found a queue family that supports both graphics and present
			queue_index_ = qfpIndex;
			break;
		}
	}
	if (queue_index_ == ~0)
	{
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
	}

	// query for Vulkan 1.3 features
	vk::StructureChain<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
		vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>
		featureChain = {
			{
				.features = {
					.sampleRateShading = vk::True,
					.samplerAnisotropy = vk::True
				}
			},
			{
				.synchronization2 = vk::True,
				.dynamicRendering = vk::True
			},
			 {
				.extendedDynamicState = vk::True
			},
			{
				.timelineSemaphore = true
			},
			{
				.shaderBufferFloat32Atomics = vk::True,
				.shaderBufferFloat32AtomicAdd = vk::True
			}
	};

	// create a Device
	float                     queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queue_index_, .queueCount = 1, .pQueuePriorities = &queuePriority };
	vk::DeviceCreateInfo      deviceCreateInfo{ 
		.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<uint32_t>(required_device_extension_.size()),
		.ppEnabledExtensionNames = required_device_extension_.data() 
	};

	device_ = vk::raii::Device(physical_device_, deviceCreateInfo);
	queue_ = vk::raii::Queue(device_, queue_index_, 0);
}

void Context::CreateCommandPool() {
	vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
										 .queueFamilyIndex = queue_index_ };
	command_pool_ = vk::raii::CommandPool(device_, poolInfo);
}

void Context::CreateCommandBuffers()
{
	// Compute
	{
		cmds_.compute.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		cmds_.compute = vk::raii::CommandBuffers(device_, allocInfo);
	}

	// Graphics
	{
		cmds_.graphics.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		cmds_.graphics = vk::raii::CommandBuffers(device_, allocInfo);
	}
}

void Context::CreateQueryPool() {
	vk::QueryPoolCreateInfo queryInfo = {};
	queryInfo.queryType = vk::QueryType::eTimestamp;
	queryInfo.queryCount = 64;

	timestamp_pool_ = device_.createQueryPool(queryInfo);
}

void Context::CreateDescriptorSetLayout()
{
	// Global UBO - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		graphics_.global_set_layout = vk::raii::DescriptorSetLayout(device_, layoutInfo);
	}

	// Object UBO + Sampler - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
		};
		counts_.ubo_dynamic += 1;
		counts_.sampler += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		graphics_.object_set_layout = vk::raii::DescriptorSetLayout(device_, layoutInfo);
	}

}

void Context::CreateDescriptorPools() {

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

	descriptor_pool_ = vk::raii::DescriptorPool(device_, poolInfo);

	// ImGUI DescriptorPool
	std::array<vk::DescriptorPoolSize, 11> imguiPoolSizes{
		vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000},
	};

	vk::DescriptorPoolCreateInfo imguiPoolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 1000,
		.poolSizeCount = static_cast<uint32_t>(imguiPoolSizes.size()),
		.pPoolSizes = imguiPoolSizes.data()
	};
	imgui_pool_ = vk::raii::DescriptorPool(device_, imguiPoolInfo);
}

void Context::CreateUniformBuffers()
{
	// Global
	{
		graphics_.global_ubo.clear();
		graphics_.global_ubo_memory.clear();
		graphics_.global_ubo_mapped = nullptr;

		auto limits = physical_device_.getProperties().limits;
		graphics_.global_slot_size = (sizeof(Graphics::GlobalUboData) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = graphics_.global_slot_size * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(physical_device_, device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		graphics_.global_ubo = std::move(buffer);
		graphics_.global_ubo_memory = std::move(bufferMem);
		graphics_.global_ubo_mapped = graphics_.global_ubo_memory.mapMemory(0, totalSize);
	}

	// Object
	{
		graphics_.object_ubo.clear();
		graphics_.object_ubo_memory.clear();
		graphics_.object_ubo_mapped = nullptr;

		auto limits = physical_device_.getProperties().limits;
		graphics_.object_slot_size = (sizeof(Graphics::ObjectUboData) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = graphics_.object_slot_size * MAX_FRAMES_IN_FLIGHT * kMaxObjects;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(physical_device_, device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		graphics_.object_ubo = std::move(buffer);
		graphics_.object_ubo_memory = std::move(bufferMem);
		graphics_.object_ubo_mapped = graphics_.object_ubo_memory.mapMemory(0, totalSize);
	}

}

void Context::CreateDescriptorSets()
{
	// Global UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*graphics_.global_set_layout
		};

		auto sets = vk::raii::DescriptorSets{ device_, allocInfo };
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
		device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Object UBO + Sampler
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*graphics_.object_set_layout
		};

		auto sets = vk::raii::DescriptorSets{ device_, allocInfo };
		graphics_.object_set = std::move(sets.front());

		vk::DescriptorBufferInfo objectUboBufferInfo{ *graphics_.object_ubo, 0, sizeof(Graphics::ObjectUboData) };
		vk::DescriptorImageInfo imageInfo{
			.sampler = *texture_->texture_sampler_,
			.imageView = *texture_->texture_image_view_,
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
		device_.updateDescriptorSets(descriptorWrites, {});
	}
}

void Context::CreateGraphicsPipelines()
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

	vk::Format depthFormat = vku::FindDepthFormat(physical_device_);

	// Model
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/model.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/model.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(device_, fragCode);

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
		auto bindingDescription = Vertex::GetBindingDescription();
		auto attributeDescriptions = Vertex::GetAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 2> setLayouts(*graphics_.global_set_layout, *graphics_.object_set_layout);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 2, .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
		graphics_.pipeline_layouts.model = vk::raii::PipelineLayout(device_, pipelineLayoutInfo);

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
		  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain_->swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
		};
		graphics_.pipelines.model = vk::raii::Pipeline(device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

}

void Context::CreateSyncObjects()
{
	in_flight_fences_.clear();

	vk::SemaphoreTypeCreateInfo semaphoreType{ .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0 };
	semaphore_ = vk::raii::Semaphore(device_, { .pNext = &semaphoreType });
	timeline_value_ = 0;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::FenceCreateInfo fenceInfo{};
		in_flight_fences_.emplace_back(device_, fenceInfo);
	}

}

void Context::CreateDepthResources() {
	vk::Format depthFormat = vku::FindDepthFormat(physical_device_);

	vku::CreateImage(physical_device_, device_, swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height, 1, msaa_samples_, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depth_image_, depth_image_memory_);
	depth_image_view_ = vku::CreateImageView(device_, depth_image_, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void Context::SetupImgui(uint32_t width, uint32_t height)
{
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.DisplaySize.x = width;
	io.DisplaySize.y = height;

	ImGui::GetStyle().FontScaleMain = 1.5f;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	VkFormat depthFmt = static_cast<VkFormat>(vku::FindDepthFormat(physical_device_));
	VkFormat colorFmt = static_cast<VkFormat>(swapchain_->swapchain_surface_format_.format);
	static VkFormat colorFormats[] = { colorFmt };
	ImGui_ImplGlfw_InitForVulkan(glfw_window_, true);
	ImGui_ImplVulkan_InitInfo init_info = {
		.ApiVersion = vk::ApiVersion14,
		.Instance = *instance_,
		.PhysicalDevice = *physical_device_,
		.Device = *device_,
		.QueueFamily = queue_index_,
		.Queue = *queue_,
		.DescriptorPool = *imgui_pool_,
		.MinImageCount = swapchain_->min_image_count_,
		.ImageCount = swapchain_->image_count_,
		.PipelineCache = NULL,
		.PipelineInfoMain = {
			.RenderPass = NULL,
			.Subpass = 0,
			.MSAASamples = static_cast<VkSampleCountFlagBits>(msaa_samples_),
			.PipelineRenderingCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
				.pNext = NULL,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &colorFmt,
				.depthAttachmentFormat = depthFmt,
				.stencilAttachmentFormat = VK_FORMAT_UNDEFINED
			},
		},
		.UseDynamicRendering = true,
		.Allocator = NULL,
	};

	ImGui_ImplVulkan_Init(&init_info);
}

