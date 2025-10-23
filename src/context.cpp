#include "swapchain.h"
#include "vulkan_utils.h"
#include "vertex.h"
#include "camera.h"
#include "model.h"
#include "texture_2d.hpp"

#include "context.h"

Context::Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height)
	: glfw_window_(glfwWindow)
{
	CreateInstance();
	SetupDebugMessenger();
	CreateSurface();
	PickPhysicalDevice();
	msaa_samples_ = GetMaxUsableSampleCount();
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
	CreateSyncObjects();

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

void Context::Draw(Camera& camera, float dt)
{
	DrawImgui();

	auto [result, imageIndex] = swapchain_->swapchain_.acquireNextImage(UINT64_MAX, nullptr, *in_flight_fences_[current_frame_]);
	while (vk::Result::eTimeout == device_.waitForFences(*in_flight_fences_[current_frame_], vk::True, UINT64_MAX))
		;
	device_.resetFences(*in_flight_fences_[current_frame_]);

	// Update timeline value for this frame
	//uint64_t computeWaitValue = timeline_value_;
	//uint64_t computeSignalValue = ++timeline_value_;
	uint64_t graphicsWaitValue = timeline_value_;
	uint64_t graphicsSignalValue = ++timeline_value_;

	{
		UpdateUniformBuffer(camera);
	}

	{

		graphics_.command_buffers[current_frame_].reset();
		graphics_.command_buffers[current_frame_].begin({});
		// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
		Transition_image_layout(
			swapchain_->swapchain_images_[imageIndex],
			graphics_.command_buffers[current_frame_],
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},                                                     // srcAccessMask (no need to wait for previous operations)
			vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
			vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
			vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
		);
		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = swapchain_->swapchain_image_views_[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
		};
		vk::RenderingInfo renderingInfo = {
			.renderArea = {.offset = { 0, 0 }, .extent = swapchain_->swapchain_extent_ },
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo
		};
		graphics_.command_buffers[current_frame_].beginRendering(renderingInfo);
		graphics_.command_buffers[current_frame_].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchain_->swapchain_extent_.width), static_cast<float>(swapchain_->swapchain_extent_.height), 0.0f, 1.0f));
		graphics_.command_buffers[current_frame_].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_->swapchain_extent_));

		{
			graphics_.command_buffers[current_frame_].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.sphere);
			graphics_.command_buffers[current_frame_].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_.pipeline_layouts.sphere, 0, *graphics_.descriptor_sets[current_frame_], nullptr);
			graphics_.command_buffers[current_frame_].bindVertexBuffers(0, { sphere_->vertex_buffer_ }, { 0 });
			graphics_.command_buffers[current_frame_].bindIndexBuffer(*sphere_->index_buffer_, 0, vk::IndexType::eUint32);
			graphics_.command_buffers[current_frame_].drawIndexed(sphere_->indices_.size(), 1, 0, 0, 0);
		}

		{
			graphics_.command_buffers[current_frame_].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.cloth);
			graphics_.command_buffers[current_frame_].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_.pipeline_layouts.cloth, 0, *graphics_.descriptor_sets[current_frame_], nullptr);
			graphics_.command_buffers[current_frame_].bindVertexBuffers(0, { particle_datas_.input }, { 0 });
			graphics_.command_buffers[current_frame_].bindIndexBuffer(*particle_datas_.index_buffer, 0, vk::IndexType::eUint32);
			graphics_.command_buffers[current_frame_].drawIndexed(particle_datas_.index_count, 1, 0, 0, 0);

		}

		// Imgui Render
		ImDrawData* draw_data = ImGui::GetDrawData();
		ImGui_ImplVulkan_RenderDrawData(draw_data, *graphics_.command_buffers[current_frame_]);

		graphics_.command_buffers[current_frame_].endRendering();

		// After rendering, transition the swapchain image to PRESENT_SRC
		Transition_image_layout(
			swapchain_->swapchain_images_[imageIndex],
			graphics_.command_buffers[current_frame_],
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
			{},                                                      // dstAccessMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
			vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
		);
		graphics_.command_buffers[current_frame_].end();

		// Submit graphics work (waits for compute to finish)
		vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;
		vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
			.waitSemaphoreValueCount = 1,
			.pWaitSemaphoreValues = &graphicsWaitValue,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = &graphicsSignalValue
		};

		vk::SubmitInfo graphicsSubmitInfo{
			.pNext = &graphicsTimelineInfo,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*semaphore_,
			.pWaitDstStageMask = &waitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &*graphics_.command_buffers[current_frame_],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*semaphore_
		};

		queue_.submit(graphicsSubmitInfo, nullptr);

		// Present the image (wait for graphics to finish)
		vk::SemaphoreWaitInfo waitInfo{
			.semaphoreCount = 1,
			.pSemaphores = &*semaphore_,
			.pValues = &graphicsSignalValue
		};

		// Wait for graphics to complete before presenting
		while (vk::Result::eTimeout == device_.waitSemaphores(waitInfo, UINT64_MAX))
			;

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
			}
			else if (result != vk::Result::eSuccess) {
				throw std::runtime_error("failed to present swap chain image!");
			}
		}
		catch (const vk::SystemError& e) {
			if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
				swapchain_->RecreateSwapChain(physical_device_, device_, surface_);
				return;
			}
			else {
				throw;
			}
		}
	}
	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Context::Transition_image_layout(
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

void Context::UpdateUniformBuffer(Camera& camera)
{
	graphics_.uniform_data.model = glm::mat4(1.0f);
	graphics_.uniform_data.view = camera.View();
	graphics_.uniform_data.proj = camera.Proj(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height);

	memcpy(graphics_.uniform_buffers_mapped[current_frame_], &graphics_.uniform_data, sizeof(graphics_.uniform_data));
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
	// graphics
	{
		graphics_.command_buffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *command_pool_;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		graphics_.command_buffers = vk::raii::CommandBuffers(device_, allocInfo);
	}

	//// compute
	//{
	//	compute_.command_buffers_.clear();
	//	vk::CommandBufferAllocateInfo allocInfo{};
	//	allocInfo.commandPool = *command_pool_;
	//	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	//	allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
	//	compute_.command_buffers_ = vk::raii::CommandBuffers(device_, allocInfo);
	//}

}

void Context::CreateDescriptorSetLayout() {
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

void Context::CreateParticleDatas()
{
	std::vector<Particle> particleBuffer(cloth_.gridsize.x * cloth_.gridsize.y);

	float dx = cloth_.size.x / (cloth_.gridsize.x - 1);
	float dy = cloth_.size.y / (cloth_.gridsize.y - 1);
	float du = 1.0f / (cloth_.gridsize.x - 1);
	float dv = 1.0f / (cloth_.gridsize.y - 1);

	// Set up a flat cloth_ that falls onto sphere
	glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(-cloth_.size.x / 2.0f, -2.0f, -cloth_.size.y / 2.0f));
	for (uint32_t i = 0; i < cloth_.gridsize.y; i++) {
		for (uint32_t j = 0; j < cloth_.gridsize.x; j++) {
			particleBuffer[i + j * cloth_.gridsize.y].pos = transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
			particleBuffer[i + j * cloth_.gridsize.y].vel = glm::vec4(0.0f);
			particleBuffer[i + j * cloth_.gridsize.y].uv = glm::vec4(1.0f - du * i, dv * j, 0.0f, 0.0f);
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
	// Graphics
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

void Context::CreateGraphicsPipelines()
{
	// Sphere
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device_, vku::ReadFile("shaders/model.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertexMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragmentMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

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
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
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
			.pStages = shaderStages,
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
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device_, vku::ReadFile("shaders/cloth.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertexMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragmentMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

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
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False
		};
		rasterizer.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisampling{
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
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
			.pStages = shaderStages,
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
			.MSAASamples = static_cast<VkSampleCountFlagBits>(vk::SampleCountFlagBits::e1),
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