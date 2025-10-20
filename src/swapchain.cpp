#include "swapchain.hpp"
#include "camera.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Swapchain::Swapchain(
	GLFWwindow* glfwWindow,
	vk::raii::Device& device,
	vk::raii::PhysicalDevice& physicalDevice,
	vk::SampleCountFlagBits msaaSamples,
	vk::raii::SurfaceKHR& surface,
	vk::raii::Queue& queue,
	vk::raii::DescriptorPool& descriptorPool,
	vk::raii::CommandPool& commandPool,
	DescriptorSetLayouts& descriptorSetLayouts
)
	: glfwWindow_(glfwWindow), device_(device), physicalDevice_(physicalDevice), msaaSamples_(msaaSamples), surface_(surface), queue_(queue), commandPool_(commandPool)
{
	stbi_set_flip_vertically_on_load(true);

	createSwapchain(physicalDevice_, surface);
	createPerImages();

	createColorResources();
	createDepthResources();

	createTextureImage();
	createTextureImageView();
	createTextureSampler();

	model = std::make_unique<Model>("models/viking_room.obj", device_, physicalDevice_, commandPool_, queue_);

	createPerFrames(descriptorPool, descriptorSetLayouts);

	createTimelineSemaphore();
}

Swapchain::~Swapchain()
{
}

void Swapchain::draw(bool& framebufferResized, Camera& camera, Pipelines& pipelines, float dt)
{
	uint32_t imageIndex = 0;
	try {
		auto [res, idx] = swapchain_.acquireNextImage(UINT64_MAX, nullptr, *frames_[currentFrame_].inFlight);
		imageIndex = idx;
		if (res == vk::Result::eSuboptimalKHR) {
			framebufferResized = false;
			recreateSwapChain();
			return;
		}
	}
	catch (const vk::OutOfDateKHRError&) {
		framebufferResized = false;
		recreateSwapChain();
		return;
	}

	while (vk::Result::eTimeout == device_.waitForFences(*frames_[currentFrame_].inFlight, vk::True, UINT64_MAX))
		;

	device_.resetFences(*frames_[currentFrame_].inFlight);

	uint64_t computeWaitValue = timelineValue_;
	uint64_t computeSignalValue = ++timelineValue_;
	uint64_t graphicsWaitValue = computeSignalValue;
	uint64_t graphicsSignalValue = ++timelineValue_;

	updateUniformBuffer(currentFrame_, camera, dt);

	// Compute
	{
		recordComputeCommandBuffer(pipelines);

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
			.pCommandBuffers = &*frames_[currentFrame_].computeCmd,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &*semaphore_
		};

		queue_.submit(computeSubmitInfo, nullptr);
	}

	// Graphics
	{
		recordGraphicsCommandBuffer(imageIndex, pipelines);

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
			.pCommandBuffers = &*frames_[currentFrame_].graphicCmd,
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
			.pSwapchains = &*swapchain_,
			.pImageIndices = &imageIndex
		};

		try {
			auto result = queue_.presentKHR(presentInfo);
		}
		catch (const vk::OutOfDateKHRError&) {
			framebufferResized = false;
			recreateSwapChain();
			return;
		}
		catch (const vk::SystemError& e) {
			// 일부 버전에선 SuboptimalKHR가 SystemError로 던져진다
			if (e.code() == vk::make_error_code(vk::Result::eSuboptimalKHR)) {
				framebufferResized = false;
				recreateSwapChain();
				return;
			}
			throw; // 다른 에러는 그대로 위로
		}

	}

	currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Swapchain::recordComputeCommandBuffer(Pipelines& pipelines)
{
	frames_[currentFrame_].computeCmd.reset();
	frames_[currentFrame_].computeCmd.begin({});
	frames_[currentFrame_].computeCmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.particleComputePipeline_);
	frames_[currentFrame_].computeCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelines.particleComputePipelineLayout_, 0, { frames_[currentFrame_].particleDescriptorSet }, {});
	frames_[currentFrame_].computeCmd.dispatch(PARTICLE_COUNT / 256, 1, 1);
	frames_[currentFrame_].computeCmd.end();
}

void Swapchain::recordGraphicsCommandBuffer(uint32_t imageIndex, Pipelines& pipelines)
{
	frames_[currentFrame_].graphicCmd.reset();
	frames_[currentFrame_].graphicCmd.begin({});

	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},                                                     // srcAccessMask (no need to wait for previous operations)
		vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
		vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
	);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

	vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = *images_[imageIndex].imageView_,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
	};
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 }, .extent = swapchainExtent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo
	};

	frames_[currentFrame_].graphicCmd.beginRendering(renderingInfo);
	frames_[currentFrame_].graphicCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.particleGraphicsPipeline_);
	frames_[currentFrame_].graphicCmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f));
	frames_[currentFrame_].graphicCmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent_));
	frames_[currentFrame_].graphicCmd.bindVertexBuffers(0, { frames_[currentFrame_].ssbo }, { 0 });
	frames_[currentFrame_].graphicCmd.draw(PARTICLE_COUNT, 1, 0, 0);

	// Imgui Render
	ImDrawData* draw_data = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(draw_data, *frames_[currentFrame_].graphicCmd);

	frames_[currentFrame_].graphicCmd.endRendering();

	// Transition the swapchain image to PRESENT_SRC
	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
		{},                                                      // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
	);

	frames_[currentFrame_].graphicCmd.end();
}

void Swapchain::transition_image_layout(
	uint32_t imageIndex,
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
		.image = images_[imageIndex].image_,
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
	frames_[currentFrame_].graphicCmd.pipelineBarrier2(dependency_info);
}

void Swapchain::transition_image_layout_custom(
	vk::raii::Image& image,
	vk::ImageLayout old_layout,
	vk::ImageLayout new_layout,
	vk::AccessFlags2 src_access_mask,
	vk::AccessFlags2 dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask,
	vk::ImageAspectFlags aspect_mask
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
	frames_[currentFrame_].graphicCmd.pipelineBarrier2(dependency_info);
}

void Swapchain::updateUniformBuffer(uint32_t currentImage, Camera& camera, float dt) {

	//// Model
	//{
	//	static auto startTime = std::chrono::high_resolution_clock::now();

	//	auto currentTime = std::chrono::high_resolution_clock::now();
	//	float time = std::chrono::duration<float>(currentTime - startTime).count();

	//	ModelUBO modelUbo{};
	//	glm::vec3 pos = { 0.0f, 0.0f, 0.0f };      // 원하는 월드 위치

	//	// updateUniformBuffer(...)
	//	glm::mat4 M = glm::mat4(1.0f);

	//	// Z-up(OBJ) -> Y-up(엔진) 보정
	//	M = glm::rotate(M, glm::radians(-90.0f), glm::vec3(1, 0, 0)); // 또는 +90.0f
	//	M = glm::rotate(M, glm::radians(-90.0f), glm::vec3(0, 0, 1)); // 또는 +90.0f

	//	M = glm::translate(M, glm::vec3(0, 0, 0));

	//	modelUbo.model = M;

	//	//modelUbo.model = glm::translate(glm::mat4(1.0f), pos);
	//	modelUbo.view = camera.view();
	//	modelUbo.proj = camera.proj(static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height));

	//	memcpy(frames_[currentImage].modelUboMapped, &modelUbo, sizeof(modelUbo));
	//}

	// Particle
	{
		ParticleUBO ubo{};
		float deltaTime = 1.0 / 60;
		ubo.deltaTime = 1.0f;

		memcpy(frames_[currentImage].particleUboMapped, &ubo, sizeof(ubo));
	}
}

void Swapchain::createSwapchain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface) {
	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	swapchainExtent_ = chooseSwapExtent(surfaceCapabilities);
	swapchainSurfaceFormat_ = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
	vk::SwapchainCreateInfoKHR swapChainCreateInfo{ .surface = surface,
													.minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
													.imageFormat = swapchainSurfaceFormat_.format,
													.imageColorSpace = swapchainSurfaceFormat_.colorSpace,
													.imageExtent = swapchainExtent_,
													.imageArrayLayers = 1,
													.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
													.imageSharingMode = vk::SharingMode::eExclusive,
													.preTransform = surfaceCapabilities.currentTransform,
													.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
													.presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
													.clipped = true };

	swapchain_ = vk::raii::SwapchainKHR(device_, swapChainCreateInfo);
}

void Swapchain::createPerImages()
{
	auto images = swapchain_.getImages();
	images_.clear();
	imageCount_ = images.size();
	images_.reserve(imageCount_);
	for (auto img : images) {
		PerImage i;
		i.image_ = img;
		i.imageView_ = createSwapchainImageView(img, swapchainSurfaceFormat_.format, device_);
		images_.emplace_back(std::move(i));
	}
}

void Swapchain::createPerFrames(
	vk::raii::DescriptorPool& descriptorPool,
	DescriptorSetLayouts& descriptorSetLayouts)
{
	createUBOs();
	createSSBOs();

	createCommandBuffers(commandPool_);

	createDescriptorSets(descriptorPool, descriptorSetLayouts);

	for (auto& f : frames_) {
		f.inFlight.clear();
		f.inFlight = vk::raii::Fence(device_, vk::FenceCreateInfo{});
	}
}

void Swapchain::createColorResources() {
	vk::Format colorFormat = swapchainSurfaceFormat_.format;

	createImage(device_, physicalDevice_, swapchainExtent_.width, swapchainExtent_.height, 1, msaaSamples_, colorFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage_, colorImageMemory_);
	colorImageView_ = createImageView(device_, colorImage_, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void Swapchain::createDepthResources() {

	vk::Format depthFormat = findDepthFormat(physicalDevice_);
	createImage(device_, physicalDevice_, swapchainExtent_.width, swapchainExtent_.height, 1, msaaSamples_, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage_, depthImageMemory_);
	depthImageView_ = createImageView(device_, depthImage_, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

uint32_t Swapchain::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
	minImageCount_ = std::max(3u, surfaceCapabilities.minImageCount);
	if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount_)) {
		minImageCount_ = surfaceCapabilities.maxImageCount;
	}
	return minImageCount_;
}

vk::SurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	assert(!availableFormats.empty());
	const auto formatIt = std::ranges::find_if(
		availableFormats,
		[](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
	return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
			// 이 함수는 디버그/프로파일링 빌드에서만 활성화되도록 #ifdef로 감싸면 더 좋습니다.
			return vk::PresentModeKHR::eImmediate;
		}
	}

	// Immediate가 없다면, 원래 로직대로 Mailbox 또는 Fifo를 선택
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
			return vk::PresentModeKHR::eMailbox;
		}
	}

	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != 0xFFFFFFFF) {
		return capabilities.currentExtent;
	}
	int width, height;
	glfwGetFramebufferSize(glfwWindow_, &width, &height);

	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
}

void Swapchain::cleanupSwapChain() {
	swapchain_ = nullptr;
}

void Swapchain::createCommandBuffers(vk::raii::CommandPool& commandPool)
{
	// graphics
	{
		vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
												 .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
		auto cmds = vk::raii::CommandBuffers(device_, allocInfo);

		for (unsigned int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			frames_[i].graphicCmd = std::move(cmds[i]);
		}
	}

	// compute
	{
		vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
												 .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
		auto cmds = vk::raii::CommandBuffers(device_, allocInfo);

		for (unsigned int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			frames_[i].computeCmd = std::move(cmds[i]);
		}
	}

}

void Swapchain::recreateSwapChain() {
	int width = 0, height = 0;
	glfwGetFramebufferSize(glfwWindow_, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(glfwWindow_, &width, &height);
		glfwWaitEvents();
	}

	device_.waitIdle();

	cleanupSwapChain();
	createSwapchain(physicalDevice_, surface_);

	images_.clear();
	createPerImages();
	createColorResources();
	createDepthResources();
}

void Swapchain::createDescriptorSets(vk::raii::DescriptorPool& descriptorPool, DescriptorSetLayouts& descriptorSetLayouts)
{
	// model
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayouts.modelDescriptorSetLayout_);
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()
		};

		auto sets = device_.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			frames_[i].modelDescriptorSet = std::move(sets[i]);

			vk::DescriptorBufferInfo bufferInfo{
				.buffer = *frames_[i].modelUbo,
				.offset = 0,
				.range = sizeof(ModelUBO)
			};
			vk::DescriptorImageInfo imageInfo{
				.sampler = *textureSampler,
				.imageView = *textureImageView,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *frames_[i].modelDescriptorSet,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &bufferInfo
				},
				vk::WriteDescriptorSet{
					.dstSet = *frames_[i].modelDescriptorSet,
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

	// Particle
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayouts.particleComputeDescriptorSetLayout_);
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()
		};

		auto sets = device_.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			frames_[i].particleDescriptorSet = std::move(sets[i]);

			vk::DescriptorBufferInfo bufferInfo{
				.buffer = *frames_[i].particleUbo,
				.offset = 0,
				.range = sizeof(ParticleUBO)
			};
			vk::DescriptorBufferInfo storageBufferInfoLastFrame(
				frames_[(i - 1) % MAX_FRAMES_IN_FLIGHT].ssbo,
				0, 
				sizeof(Particle)* PARTICLE_COUNT
			);
			vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(
				frames_[i].ssbo,
				0, 
				sizeof(Particle)* PARTICLE_COUNT
			);
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *frames_[i].particleDescriptorSet, 
					.dstBinding = 0, 
					.dstArrayElement = 0, 
					.descriptorCount = 1, 
					.descriptorType = vk::DescriptorType::eUniformBuffer, 
					.pImageInfo = nullptr, 
					.pBufferInfo = &bufferInfo, 
					.pTexelBufferView = nullptr 
				},
				vk::WriteDescriptorSet{
					.dstSet = *frames_[i].particleDescriptorSet,
					.dstBinding = 1, 
					.dstArrayElement = 0, 
					.descriptorCount = 1, 
					.descriptorType = vk::DescriptorType::eStorageBuffer, 
					.pImageInfo = nullptr, 
					.pBufferInfo = &storageBufferInfoLastFrame, 
					.pTexelBufferView = nullptr 
				},
				vk::WriteDescriptorSet{
					.dstSet = *frames_[i].particleDescriptorSet,
					.dstBinding = 2, 
					.dstArrayElement = 0, 
					.descriptorCount = 1, 
					.descriptorType = vk::DescriptorType::eStorageBuffer, 
					.pImageInfo = nullptr, 
					.pBufferInfo = &storageBufferInfoCurrentFrame, 
					.pTexelBufferView = nullptr 
				},
			};
			device_.updateDescriptorSets(descriptorWrites, {});
		}
	}
}

void Swapchain::createTextureImage() {
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("models/viking_room.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	vk::DeviceSize imageSize = texWidth * texHeight * 4;
	mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	vk::raii::Buffer stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	createBuffer(device_, physicalDevice_, imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* data = stagingBufferMemory.mapMemory(0, imageSize);
	memcpy(data, pixels, imageSize);
	stagingBufferMemory.unmapMemory();

	stbi_image_free(pixels);

	createImage(device_, physicalDevice_, texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

	transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
	copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

	generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
}

void Swapchain::generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
	// Check if image format supports linear blit-ing
	vk::FormatProperties formatProperties = physicalDevice_.getFormatProperties(imageFormat);

	if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
		throw std::runtime_error("texture image format does not support linear blitting!");
	}

	std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();

	vk::ImageMemoryBarrier barrier = { .srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eTransferRead
						   , .oldLayout = vk::ImageLayout::eTransferDstOptimal, .newLayout = vk::ImageLayout::eTransferSrcOptimal
						   , .srcQueueFamilyIndex = vk::QueueFamilyIgnored, .dstQueueFamilyIndex = vk::QueueFamilyIgnored, .image = image };
	barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.levelCount = 1;

	int32_t mipWidth = texWidth;
	int32_t mipHeight = texHeight;

	for (uint32_t i = 1; i < mipLevels; i++) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

		vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
		offsets[0] = vk::Offset3D(0, 0, 0);
		offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
		dstOffsets[0] = vk::Offset3D(0, 0, 0);
		dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);
		vk::ImageBlit blit = { .srcSubresource = {}, .srcOffsets = offsets,
							.dstSubresource = {}, .dstOffsets = dstOffsets };
		blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
		blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

		commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

		barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
		barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}

	barrier.subresourceRange.baseMipLevel = mipLevels - 1;
	barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
	barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

	commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

	endSingleTimeCommands(*commandBuffer);
}

void Swapchain::copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
	std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
	vk::BufferImageCopy region{ .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
							   .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
							   .imageOffset = {0, 0, 0}, .imageExtent = {width, height, 1} };
	commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
	endSingleTimeCommands(*commandBuffer);
}

void Swapchain::transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
	auto commandBuffer = beginSingleTimeCommands();

	vk::ImageMemoryBarrier barrier{
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.image = image,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1}
	};
	vk::PipelineStageFlags sourceStage;
	vk::PipelineStageFlags destinationStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

		sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStage = vk::PipelineStageFlagBits::eTransfer;
	}
	else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		sourceStage = vk::PipelineStageFlagBits::eTransfer;
		destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
	}
	else {
		throw std::invalid_argument("unsupported layout transition!");
	}
	commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
	endSingleTimeCommands(*commandBuffer);
}

std::unique_ptr<vk::raii::CommandBuffer> Swapchain::beginSingleTimeCommands() {
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool_, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
	std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device_, allocInfo).front()));

	vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	commandBuffer->begin(beginInfo);

	return commandBuffer;
}

void Swapchain::endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
	commandBuffer.end();

	vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
	queue_.submit(submitInfo, nullptr);
	queue_.waitIdle();
}

void Swapchain::createTextureImageView() {
	textureImageView = createImageView(device_, textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
}

void Swapchain::createTextureSampler() {
	vk::PhysicalDeviceProperties properties = physicalDevice_.getProperties();
	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::True,
		.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways };
	textureSampler = vk::raii::Sampler(device_, samplerInfo);
}

void Swapchain::createUBOs()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

		// Model
		{
			frames_[i].modelUbo.clear();
			frames_[i].modelUboMem.clear();
			frames_[i].modelUboMapped = nullptr;

			vk::DeviceSize bufferSize = sizeof(ModelUBO);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);

			frames_[i].modelUbo = std::move(buffer);
			frames_[i].modelUboMem = std::move(bufferMem);
			frames_[i].modelUboMapped = frames_[i].modelUboMem.mapMemory(0, bufferSize);
		}

		// Particle
		{
			frames_[i].particleUbo.clear();
			frames_[i].particleUboMem.clear();
			frames_[i].particleUboMapped = nullptr;

			vk::DeviceSize bufferSize = sizeof(ParticleUBO);
			vk::raii::Buffer buffer({});
			vk::raii::DeviceMemory bufferMem({});
			createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);

			frames_[i].particleUbo = std::move(buffer);
			frames_[i].particleUboMem = std::move(bufferMem);
			frames_[i].particleUboMapped = frames_[i].particleUboMem.mapMemory(0, bufferSize);
		}
	}
}

void Swapchain::createSSBOs()
{
	// Initialize particles
	std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
	std::uniform_real_distribution rndDist(0.0f, 1.0f);

	// Initial particle positions on a circle
	std::vector<Particle> particles(PARTICLE_COUNT);
	for (auto& particle : particles) {
		float r = 0.25f * sqrtf(rndDist(rndEngine));
		float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
		float x = r * cosf(theta) * swapchainExtent_.width / swapchainExtent_.height;
		float y = r * sinf(theta);
		particle.position = glm::vec2(x, y);
		particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
		particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
	}

	vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

	// Create a staging buffer used to upload data to the gpu
	vk::raii::Buffer stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, particles.data(), (size_t)bufferSize);
	stagingBufferMemory.unmapMemory();

	// Copy initial particle data to all storage buffers
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		frames_[i].ssbo.clear();
		frames_[i].ssboMem.clear();

		vk::raii::Buffer shaderStorageBufferTemp({});
		vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
		createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
		copyBuffer(device_, commandPool_, stagingBuffer, shaderStorageBufferTemp, bufferSize, queue_);
		frames_[i].ssbo = std::move(shaderStorageBufferTemp);
		frames_[i].ssboMem = std::move(shaderStorageBufferTempMemory);
	}
}

vk::raii::ImageView Swapchain::createSwapchainImageView(vk::Image& image, vk::Format format, vk::raii::Device& device)
{
	vk::ImageViewCreateInfo imageViewCreateInfo{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
	  .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
	};
	return vk::raii::ImageView{ device, imageViewCreateInfo };
}

void Swapchain::createTimelineSemaphore()
{
	vk::SemaphoreTypeCreateInfo semaphoreType{ .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0 };
	semaphore_ = vk::raii::Semaphore(device_, { .pNext = &semaphoreType });
	timelineValue_ = 0;
}