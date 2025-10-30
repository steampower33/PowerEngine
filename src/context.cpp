#include "swapchain.h"
#include "vulkan_utils.h"
#include "vertex.h"
#include "camera.h"
#include "model.h"
#include "texture_2d.h"
#include "mouse_interactor.h"

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

	{
		sphere_ = std::make_unique<Model>("assets/models/sphere.gltf", physical_device_, device_, queue_, command_pool_);

		texture_ = std::make_unique<Texture2D>("assets/textures/vulkan_cloth_rgba.ktx", physical_device_, device_, queue_, command_pool_);
	}

	CreateDescriptorSetLayout();
	CreateDescriptorPools();

	CreateUniformBuffers();
	CreateParticleDatas();

	CreateDescriptorSets();
	CreateGraphicsPipelines();
	CreateComputePipelines();
	CreateSyncObjects();

	CreateDepthResources();

	SetupImgui(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height);
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

	sphere_->UpdateUBO(camera, glm::vec2(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height), current_frame_);

	UpdateComputeUBO();
	UpdateGraphicsUBO(camera);
}

void Context::Draw()
{
	DrawImgui();

	auto [result, imageIndex] = swapchain_->swapchain_.acquireNextImage(UINT64_MAX, nullptr, in_flight_fences_[current_frame_]);

	while (vk::Result::eTimeout == device_.waitForFences(*in_flight_fences_[current_frame_], vk::True, UINT64_MAX));
	device_.resetFences(*in_flight_fences_[current_frame_]);

	uint64_t computeWaitValue = timeline_value_;
	uint64_t computeSignalValue = ++timeline_value_;
	uint64_t graphicsWaitValue = computeSignalValue;
	uint64_t graphicsSignalValue = ++timeline_value_;

	RecordMassSpringComputeCommandBuffer();
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
			.pCommandBuffers = &*compute_.command_buffers[current_frame_],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*semaphore_
		};

		queue_.submit(computeSubmitInfo, nullptr);
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
			.pCommandBuffers = &*graphics_.command_buffers[current_frame_],
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

// vk::Image Transition
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

// vk::raii:Image Transition
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
		ImGui::End();
	}

	ImGui::Render();
}

void Context::UpdateMouseInteractor(Camera& camera, MouseInteractor& mouse_interactor)
{
	mouse_interactor.Update(camera, glm::vec2(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height), *sphere_);
}

void Context::UpdateComputeUBO()
{
	compute_.uniform_data.deltaT = 0.00001f;
	compute_.uniform_data.springStiffness = 1000.0f;
	compute_.uniform_data.spherePos = glm::vec4(sphere_->position_, 0.0f);

	memcpy(compute_.uniform_buffers_mapped[current_frame_], &compute_.uniform_data, sizeof(Compute::UniformData));
}

void Context::RecordMassSpringComputeCommandBuffer()
{
	const auto& cmd = compute_.command_buffers[current_frame_];

	cmd.begin({});

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *compute_.pipeline);

	// Dispatch the compute job
	const uint32_t iterations = 64;
	for (uint32_t j = 0; j < iterations; j++) {
		// 핑퐁: 디스크립터 셋을 번갈아 바인딩
		read_set_ = 1 - read_set_; // read_set_은 Context 클래스의 멤버 변수여야 합니다.
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,
			*compute_.pipeline_layout,
			0, // firstSet
			{ *compute_.descriptor_sets[read_set_] }, // 바인딩할 셋
			nullptr // dynamicOffsets
		);

		// 마지막 반복에서만 법선을 계산하라는 신호를 푸시 콘스턴트로 보냄
		uint32_t calculateNormals = (j == iterations - 1) ? 1 : 0;
		cmd.pushConstants<uint32_t>(
			*compute_.pipeline_layout,
			vk::ShaderStageFlagBits::eCompute,
			0, // offset
			calculateNormals
		);

		// 컴퓨트 작업 실행
		cmd.dispatch(cloth_.gridsize.x / 10, cloth_.gridsize.y / 10, 1);

		// Don't add a barrier on the last iteration
		if (j != iterations - 1) {
			// 이번 Dispatch에서 쓴 버퍼를 다음 Dispatch에서 읽기 전에 동기화
			vk::Buffer bufferToBarrier = (read_set_ == 0) ? *particle_datas_.input : *particle_datas_.output;
			AddComputeToComputeBarrier(cmd, bufferToBarrier);
		}
	}

	cmd.end();
}

void Context::AddComputeToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer)
{
	vk::BufferMemoryBarrier2 bufferBarrier{
		.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite, // 이전 작업: 셰이더 쓰기
		.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.dstAccessMask = vk::AccessFlagBits2::eShaderRead,  // 다음 작업: 셰이더 읽기
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = buffer,
		.offset = 0,
		.size = VK_WHOLE_SIZE
	};
	vk::DependencyInfo dependencyInfo{ .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &bufferBarrier };
	cmd.pipelineBarrier2(dependencyInfo);
}

// 그래픽스 -> 컴퓨트 큐로 자원 소유권 이전 (Acquire)
void Context::AddGraphicsToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer)
{
	vk::BufferMemoryBarrier2 bufferBarrier{
		.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
		.srcAccessMask = {}, // 이전 접근 정보 필요 없음 (소유권만 이전)
		.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // 그래픽스 큐에서
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // 컴퓨트 큐로
		.buffer = buffer,
		.offset = 0,
		.size = VK_WHOLE_SIZE
	};
	vk::DependencyInfo dependencyInfo{ .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &bufferBarrier };
	cmd.pipelineBarrier2(dependencyInfo);
}

// 컴퓨트 -> 그래픽스 큐로 자원 소유권 이전 (Release)
void Context::AddComputeToGraphicsBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer)
{
	vk::BufferMemoryBarrier2 bufferBarrier{
		.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
		.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
		.dstAccessMask = {}, // 다음 접근 정보 필요 없음 (소유권만 이전)
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // 컴퓨트 큐에서
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // 그래픽스 큐로
		.buffer = buffer,
		.offset = 0,
		.size = VK_WHOLE_SIZE
	};
	vk::DependencyInfo dependencyInfo{ .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &bufferBarrier };
	cmd.pipelineBarrier2(dependencyInfo);
}

void Context::UpdateGraphicsUBO(Camera& camera)
{
	graphics_.uniform_data.model = glm::mat4(1.0f);
	graphics_.uniform_data.view = camera.View();
	graphics_.uniform_data.proj = camera.Proj(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height);

	memcpy(graphics_.uniform_buffers_mapped[current_frame_], &graphics_.uniform_data, sizeof(graphics_.uniform_data));
}

void Context::RecordGraphicsCommandBuffer(uint32_t imageIndex)
{
	const auto& cmd = graphics_.command_buffers[current_frame_];

	cmd.reset();
	cmd.begin({});

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
	cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchain_->swapchain_extent_.width), static_cast<float>(swapchain_->swapchain_extent_.height), 0.0f, 1.0f));
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_->swapchain_extent_));

	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.sphere);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_.pipeline_layouts.sphere, 0, *sphere_->descriptor_sets[current_frame_], nullptr);
		cmd.bindVertexBuffers(0, { sphere_->vertex_buffer_ }, { 0 });
		cmd.bindIndexBuffer(*sphere_->index_buffer_, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(sphere_->indices_.size(), 1, 0, 0, 0);
	}

	{
		vk::Buffer finalResultBuffer = (read_set_ == 0) ? *particle_datas_.input : *particle_datas_.output;
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.cloth);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_.pipeline_layouts.cloth, 0, *graphics_.descriptor_sets[current_frame_], nullptr);
		cmd.bindVertexBuffers(0, { finalResultBuffer }, { 0 });
		cmd.bindIndexBuffer(*particle_datas_.index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(particle_datas_.index_count, 1, 0, 0, 0);
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

	// Check if the required layers are supported by the Vulkan implementation.
	auto layerProperties = context_.enumerateInstanceLayerProperties();
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
	auto extensionProperties = context_.enumerateInstanceExtensionProperties();
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
	instance_ = vk::raii::Instance(context_, createInfo);
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
			bool supportsAllRequiredExtensions =
				std::ranges::all_of(required_device_extension_,
					[&availableDeviceExtensions](auto const& requiredDeviceExtension)
					{
						return std::ranges::any_of(availableDeviceExtensions,
							[requiredDeviceExtension](auto const& availableDeviceExtension)
							{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
					});

			auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2,
				vk::PhysicalDeviceVulkan13Features,
				vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
				vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
			bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
				features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
				features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
				features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;

			return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
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
		vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
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
			}
	};

	// create a Device
	float                     queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queue_index_, .queueCount = 1, .pQueuePriorities = &queuePriority };
	vk::DeviceCreateInfo      deviceCreateInfo{ .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
												.queueCreateInfoCount = 1,
												.pQueueCreateInfos = &deviceQueueCreateInfo,
												.enabledExtensionCount = static_cast<uint32_t>(required_device_extension_.size()),
												.ppEnabledExtensionNames = required_device_extension_.data() };

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
	// Graphics
	{
		graphics_.command_buffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		graphics_.command_buffers = vk::raii::CommandBuffers(device_, allocInfo);
	}

	// Compute
	{
		compute_.command_buffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		compute_.command_buffers = vk::raii::CommandBuffers(device_, allocInfo);
	}

}

void Context::CreateDescriptorSetLayout()
{
	// Model
	{
		std::array layoutBindings{
		vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
		};
		counts_.ubo += 1;
		counts_.sampler += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		sphere_->descriptor_set_layout = vk::raii::DescriptorSetLayout(device_, layoutInfo);

	}

	// Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
		};
		counts_.ubo += 1;
		counts_.sampler += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		graphics_.descriptor_set_layout = vk::raii::DescriptorSetLayout(device_, layoutInfo);
	}

	// Compute
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
			vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
			vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
		};
		counts_.ubo += 1 * 2;
		counts_.sb += 2 * 2;
		counts_.layout += 2;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		compute_.descriptor_set_layout = vk::raii::DescriptorSetLayout(device_, layoutInfo);
	}
}

void Context::CreateDescriptorPools() {

	std::vector<vk::DescriptorPoolSize> poolSizes;

	if (counts_.ubo > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * counts_.ubo);
	}
	if (counts_.sampler > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * counts_.sampler);
	}
	if (counts_.sb > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * counts_.sb);
	}

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = counts_.layout * MAX_FRAMES_IN_FLIGHT,
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
	//Models
	{
		sphere_->uniform_buffers.clear();
		sphere_->uniform_buffers_memory.clear();
		sphere_->uniform_buffers_mapped.clear();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DeviceSize bufferSize = sizeof(Graphics::UniformData);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
			sphere_->uniform_buffers.emplace_back(std::move(buffer));
			sphere_->uniform_buffers_memory.emplace_back(std::move(bufferMem));
			sphere_->uniform_buffers_mapped.emplace_back(sphere_->uniform_buffers_memory[i].mapMemory(0, bufferSize));
		}

	}

	// Graphics
	{
		graphics_.uniform_buffers.clear();
		graphics_.uniform_buffers_memory.clear();
		graphics_.uniform_buffers_mapped.clear();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DeviceSize bufferSize = sizeof(Graphics::UniformData);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
			graphics_.uniform_buffers.emplace_back(std::move(buffer));
			graphics_.uniform_buffers_memory.emplace_back(std::move(bufferMem));
			graphics_.uniform_buffers_mapped.emplace_back(graphics_.uniform_buffers_memory[i].mapMemory(0, bufferSize));
		}
	}

	// Compute
	{
		// Set some initial values
		float dx = cloth_.size.x / (cloth_.gridsize.x - 1);
		float dy = cloth_.size.y / (cloth_.gridsize.y - 1);

		compute_.uniform_data.restDistH = dx;
		compute_.uniform_data.restDistV = dy;
		compute_.uniform_data.restDistD = sqrtf(dx * dx + dy * dy);
		compute_.uniform_data.particleCount = cloth_.gridsize;

		compute_.uniform_buffers.clear();
		compute_.uniform_buffers_memory.clear();
		compute_.uniform_buffers_mapped.clear();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DeviceSize bufferSize = sizeof(Compute::UniformData);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
			compute_.uniform_buffers.emplace_back(std::move(buffer));
			compute_.uniform_buffers_memory.emplace_back(std::move(bufferMem));
			compute_.uniform_buffers_mapped.emplace_back(compute_.uniform_buffers_memory[i].mapMemory(0, bufferSize));
		}
	}
}

void Context::CreateParticleDatas()
{
	std::vector<Particle> particleBuffer(cloth_.gridsize.x * cloth_.gridsize.y);

	float dx = cloth_.size.x / (cloth_.gridsize.x - 1);
	float dy = cloth_.size.y / (cloth_.gridsize.y - 1);
	float du = 1.0f / (cloth_.gridsize.x - 1);
	float dv = 1.0f / (cloth_.gridsize.y - 1);

	// Set up a flat cloth_ that falls onto sphere
	glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(-cloth_.size.x / 2.0f, 2.0f, -cloth_.size.y / 2.0f));
	for (uint32_t i = 0; i < cloth_.gridsize.y; i++) {
		for (uint32_t j = 0; j < cloth_.gridsize.x; j++) {
			uint32_t index = i * cloth_.gridsize.x + j;
			particleBuffer[index].pos = transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
			particleBuffer[index].vel = glm::vec4(0.0f);
			particleBuffer[index].uv = glm::vec4(du * j, dv * i, 0.0f, 0.0f);
			particleBuffer[index].normal = glm::vec4(0.0f);
		}
	}

	vk::DeviceSize bufferSize = sizeof(Particle) * particleBuffer.size();

	// Create a staging buffer used to upload data to the gpu
	vk::raii::Buffer stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, particleBuffer.data(), (size_t)bufferSize);
	stagingBufferMemory.unmapMemory();

	{
		particle_datas_.input.clear();
		particle_datas_.input_memory.clear();

		vk::raii::Buffer shaderStorageBufferTemp({});
		vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
		vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
		vku::CopyBuffer(device_, queue_, command_pool_, stagingBuffer, shaderStorageBufferTemp, bufferSize);
		particle_datas_.input = std::move(shaderStorageBufferTemp);
		particle_datas_.input_memory = std::move(shaderStorageBufferTempMemory);
	}

	{
		particle_datas_.output.clear();
		particle_datas_.output_memory.clear();

		vk::raii::Buffer shaderStorageBufferTemp({});
		vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
		vku::CreateBuffer(physical_device_, device_, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
		vku::CopyBuffer(device_, queue_, command_pool_, stagingBuffer, shaderStorageBufferTemp, bufferSize);
		particle_datas_.output = std::move(shaderStorageBufferTemp);
		particle_datas_.output_memory = std::move(shaderStorageBufferTempMemory);
	}

	// Indices
	std::vector<uint32_t> indices;
	for (uint32_t y = 0; y < cloth_.gridsize.y - 1; y++) {
		for (uint32_t x = 0; x < cloth_.gridsize.x; x++) {
			indices.push_back((y + 1) * cloth_.gridsize.x + x);
			indices.push_back((y)*cloth_.gridsize.x + x);
		}
		// Primitive restart (signaled by special value 0xFFFFFFFF)
		indices.push_back(0xFFFFFFFF);
	}
	uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(uint32_t);
	particle_datas_.index_count = static_cast<uint32_t>(indices.size());

	vku::CreateIndexBuffer(physical_device_, device_, queue_, command_pool_, indices, particle_datas_.index_buffer, particle_datas_.index_buffer_memory);
}

void Context::CreateDescriptorSets()
{
	// Models
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, sphere_->descriptor_set_layout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.descriptorPool = *descriptor_pool_;
		allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
		allocInfo.pSetLayouts = layouts.data();
		sphere_->descriptor_sets.clear();
		sphere_->descriptor_sets = device_.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DescriptorBufferInfo bufferInfo(
				sphere_->uniform_buffers[i],
				0,
				sizeof(Graphics::UniformData)
			);
			vk::DescriptorImageInfo imageInfo{
				.sampler = texture_->texture_sampler_,
				.imageView = texture_->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array descriptorWrites{
					vk::WriteDescriptorSet{
						.dstSet = sphere_->descriptor_sets[i],
						.dstBinding = 0,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eUniformBuffer,
						.pBufferInfo = &bufferInfo
					},
					vk::WriteDescriptorSet{
						.dstSet = sphere_->descriptor_sets[i],
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

	// Graphics
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphics_.descriptor_set_layout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.descriptorPool = *descriptor_pool_;
		allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
		allocInfo.pSetLayouts = layouts.data();
		graphics_.descriptor_sets.clear();
		graphics_.descriptor_sets = device_.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DescriptorBufferInfo bufferInfo(
				graphics_.uniform_buffers[i],
				0,
				sizeof(Graphics::UniformData)
			);
			vk::DescriptorImageInfo imageInfo{
				.sampler = texture_->texture_sampler_,
				.imageView = texture_->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array descriptorWrites{
					vk::WriteDescriptorSet{
						.dstSet = graphics_.descriptor_sets[i],
						.dstBinding = 0,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eUniformBuffer,
						.pBufferInfo = &bufferInfo
					},
					vk::WriteDescriptorSet{
						.dstSet = graphics_.descriptor_sets[i],
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

	// Compute
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, compute_.descriptor_set_layout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.descriptorPool = *descriptor_pool_;
		allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
		allocInfo.pSetLayouts = layouts.data();
		compute_.descriptor_sets.clear();
		compute_.descriptor_sets = device_.allocateDescriptorSets(allocInfo);

		{
			vk::DescriptorBufferInfo inputInfo(particle_datas_.input, 0, sizeof(Particle) * cloth_.gridsize.x * cloth_.gridsize.y);
			vk::DescriptorBufferInfo outputInfo(particle_datas_.output, 0, sizeof(Particle) * cloth_.gridsize.x * cloth_.gridsize.y);
			vk::DescriptorBufferInfo bufferInfo(
				compute_.uniform_buffers[0],
				0,
				sizeof(Compute::UniformData)
			);
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[0],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &inputInfo,
					.pTexelBufferView = nullptr
				},
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[0],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &outputInfo,
					.pTexelBufferView = nullptr
				},
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[0],
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &bufferInfo,
					.pTexelBufferView = nullptr
				}
			};
			device_.updateDescriptorSets(descriptorWrites, {});
		}

		{
			vk::DescriptorBufferInfo inputInfo(particle_datas_.output, 0, sizeof(Particle) * cloth_.gridsize.x * cloth_.gridsize.y);
			vk::DescriptorBufferInfo outputInfo(particle_datas_.input, 0, sizeof(Particle) * cloth_.gridsize.x * cloth_.gridsize.y);
			vk::DescriptorBufferInfo bufferInfo(
				compute_.uniform_buffers[1],
				0,
				sizeof(Compute::UniformData)
			);
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[1],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &inputInfo,
					.pTexelBufferView = nullptr
				},
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[1],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &outputInfo,
					.pTexelBufferView = nullptr
				},
				vk::WriteDescriptorSet{
					.dstSet = *compute_.descriptor_sets[1],
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pImageInfo = nullptr,
					.pBufferInfo = &bufferInfo,
					.pTexelBufferView = nullptr
				}
			};
			device_.updateDescriptorSets(descriptorWrites, {});
		}
	}
}

void Context::CreateGraphicsPipelines()
{
	// Sphere
	{
		auto vertCode = vku::ReadFile("shaders/model.vert.spv");
		auto fragCode = vku::ReadFile("shaders/model.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(device_, fragCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"              // GLSL 기본 엔트리
		};
		vk::PipelineShaderStageCreateInfo fragStage{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = *fragModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages{ vertStage, fragStage };

		auto bindingDescription = Vertex::GetBindingDescription();
		auto attributeDescriptions = Vertex::GetAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};
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

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*graphics_.descriptor_set_layout, .pushConstantRangeCount = 0 };

		graphics_.pipeline_layouts.sphere = vk::raii::PipelineLayout(device_, pipelineLayoutInfo);

		vk::Format depthFormat = vku::FindDepthFormat(physical_device_);

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
			.layout = graphics_.pipeline_layouts.sphere,
			.renderPass = nullptr },
		  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain_->swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
		};

		graphics_.pipelines.sphere = vk::raii::Pipeline(device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Cloth
	{
		auto vertCode = vku::ReadFile("shaders/cloth.vert.spv");
		auto fragCode = vku::ReadFile("shaders/cloth.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(device_, fragCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"              // GLSL 기본 엔트리
		};
		vk::PipelineShaderStageCreateInfo fragStage{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = *fragModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages{ vertStage, fragStage };

		auto bindingDescription = Particle::GetBindingDescription();
		auto attributeDescriptions = Particle::GetAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
			.topology = vk::PrimitiveTopology::eTriangleStrip,
			.primitiveRestartEnable = vk::True
		};
		vk::PipelineViewportStateCreateInfo viewportState{
			.viewportCount = 1,
			.scissorCount = 1
		};
		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
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

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*graphics_.descriptor_set_layout, .pushConstantRangeCount = 0 };

		graphics_.pipeline_layouts.cloth = vk::raii::PipelineLayout(device_, pipelineLayoutInfo);

		vk::Format depthFormat = vku::FindDepthFormat(physical_device_);

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
			.layout = graphics_.pipeline_layouts.cloth,
			.renderPass = nullptr },
		  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain_->swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
		};

		graphics_.pipelines.cloth = vk::raii::Pipeline(device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

}

void Context::CreateComputePipelines()
{
	auto compCode = vku::ReadFile("shaders/cloth.comp.spv");
	vk::raii::ShaderModule compModule = vku::CreateShaderModule(device_, compCode);

	vk::PushConstantRange pushConstantRange{
		.stageFlags = vk::ShaderStageFlagBits::eCompute,
		.offset = 0,
		.size = sizeof(uint32_t)
	};
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*compute_.descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange
	};
	compute_.pipeline_layout = vk::raii::PipelineLayout(device_, pipelineLayoutInfo);
	vk::ComputePipelineCreateInfo pipelineInfo{ 
		.stage = vk::PipelineShaderStageCreateInfo{
			.stage = vk::ShaderStageFlagBits::eCompute,
			.module = *compModule,
			.pName = "main"          // GLSL compute 엔트리
		}, 
		.layout = *compute_.pipeline_layout };
	compute_.pipeline = vk::raii::Pipeline(device_, nullptr, pipelineInfo);
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

