#include "swapchain.hpp"
#include "camera.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Swapchain::Swapchain(
	GLFWwindow* glfwWindow,
	vk::raii::Device& device,
	vk::raii::PhysicalDevice& physicalDevice,
	vk::raii::SurfaceKHR& surface,
	uint32_t queueFamilyIndex,
	vk::raii::CommandPool& commandPool,
	vk::raii::DescriptorSetLayout& descriptorSetLayout,
	vk::raii::DescriptorPool& descriptorPool,
	vk::raii::Queue& queue)
	: glfwWindow_(glfwWindow), device_(device), physicalDevice_(physicalDevice), surface_(surface), commandPool_(commandPool), queue_(queue)
{
	stbi_set_flip_vertically_on_load(true);

	createSwapchain(physicalDevice_, surface);
	createPerImages();

	createTextureImage();
	createTextureImageView();
	createTextureSampler();

	createVertexBuffer();
	createIndexBuffer();

	createPerFrames(descriptorSetLayout, descriptorPool);
}

Swapchain::~Swapchain()
{
}

void Swapchain::draw(bool& framebufferResized, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout, Camera& camera)
{
	while (vk::Result::eTimeout == device_.waitForFences(*frames_[currentFrame_].inFlight, vk::True, UINT64_MAX))
		;
	uint32_t imageIndex = 0;
	try {
		auto [res, idx] = swapchain_.acquireNextImage(
			UINT64_MAX, *frames_[currentFrame_].imageAvailable, nullptr);
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

	if (auto f = images_[imageIndex].lastSubmitFence;
		f != VK_NULL_HANDLE)
	{
		// vk-hpp는 ArrayProxy 오버로드라 단일 펜스도 이렇게 호출 가능
		vk::Result r = device_.waitForFences(f, vk::True, UINT64_MAX);
		// 보통 eSuccess 여야 함. (UINT64_MAX면 timeout 날 일이 사실상 없음)
		if (r != vk::Result::eSuccess) {
			// 필요시 로깅/예외/복구 로직
			throw std::runtime_error("waitForFences failed: " + vk::to_string(r));
		}
	}
	images_[imageIndex].lastSubmitFence = frames_[currentFrame_].inFlight;

	updateUniformBuffer(currentFrame_, camera);

	device_.resetFences(*frames_[currentFrame_].inFlight);

	frames_[currentFrame_].cmd.reset();
	recordCommandBuffer(imageIndex, queue, graphicsPipeline, pipelineLayout);

	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*frames_[currentFrame_].imageAvailable,
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*frames_[currentFrame_].cmd,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*images_[imageIndex].renderFinished_ };
	queue.submit(submitInfo, *frames_[currentFrame_].inFlight);

	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*images_[imageIndex].renderFinished_,
		.swapchainCount = 1,
		.pSwapchains = &*swapchain_,
		.pImageIndices = &imageIndex };
	try {
		queue.presentKHR(presentInfoKHR);   // 실패 시 예외 발생 (리턴값 없음)
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

	currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Swapchain::recordCommandBuffer(uint32_t imageIndex, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout) {
	frames_[currentFrame_].cmd.begin({});
	// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},                                                     // srcAccessMask (no need to wait for previous operations)
		vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
		vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
	);

	// Transition depth image to depth attachment optimal layout
	vk::ImageMemoryBarrier2 depthBarrier = {
		.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
		.srcAccessMask = {},
		.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		.oldLayout = vk::ImageLayout::eUndefined,
		.newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = images_[imageIndex].depthImage_,
		.subresourceRange = {
			.aspectMask = vk::ImageAspectFlagBits::eDepth,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk::DependencyInfo depthDependencyInfo = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &depthBarrier
	};
	frames_[currentFrame_].cmd.pipelineBarrier2(depthDependencyInfo);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

	vk::RenderingAttachmentInfo colorAttachmentInfo = {
			.imageView = *images_[imageIndex].imageView_,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
	};

	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = *images_[imageIndex].depthImageView_,
		.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};

	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 }, .extent = swapchainExtent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	frames_[currentFrame_].cmd.beginRendering(renderingInfo);
	frames_[currentFrame_].cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
	frames_[currentFrame_].cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f));
	frames_[currentFrame_].cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent_));
	frames_[currentFrame_].cmd.bindVertexBuffers(0, *vertexBuffer_, { 0 });
	frames_[currentFrame_].cmd.bindIndexBuffer(indexBuffer_, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
	frames_[currentFrame_].cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *frames_[currentFrame_].descriptorSet, nullptr);
	frames_[currentFrame_].cmd.drawIndexed(indices.size(), 1, 0, 0, 0);
	frames_[currentFrame_].cmd.endRendering();
	// After rendering, transition the swapchain image to PRESENT_SRC
	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
		{},                                                      // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
	);
	frames_[currentFrame_].cmd.end();
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
	frames_[currentFrame_].cmd.pipelineBarrier2(dependency_info);
}

void Swapchain::updateUniformBuffer(uint32_t currentImage, Camera& camera) {
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float>(currentTime - startTime).count();

	UniformBufferObject ubo{};
	glm::vec3 pos = { 0.0f, 0.0f, 0.0f };      // 원하는 월드 위치
	ubo.model = glm::translate(glm::mat4(1.0f), pos);
	ubo.view = camera.view();
	ubo.proj = camera.proj(static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height));

	memcpy(frames_[currentImage].uboMapped, &ubo, sizeof(ubo));
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
	images_.reserve(images.size());
	for (auto img : images) {
		PerImage i;
		i.image_ = img;
		i.imageView_ = createSwapchainImageView(img, swapchainSurfaceFormat_.format, device_);

		vk::Format depthFormat = findDepthFormat(physicalDevice_);
		createImage(device_, physicalDevice_, swapchainExtent_.width, swapchainExtent_.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, i.depthImage_, i.depthMem_);
		i.depthImageView_ = createImageView(device_, i.depthImage_, depthFormat, vk::ImageAspectFlagBits::eDepth);

		i.renderFinished_ = vk::raii::Semaphore(device_, vk::SemaphoreCreateInfo());

		i.lastSubmitFence = VK_NULL_HANDLE;
		images_.emplace_back(std::move(i));
	}
}

void Swapchain::createPerFrames(
	vk::raii::DescriptorSetLayout& descriptorSetLayout,
	vk::raii::DescriptorPool& descriptorPool)
{
	createUniformBuffers();
	createDescriptorSets(descriptorSetLayout, descriptorPool);
	createCommandBuffers(commandPool_);

	for (auto& f : frames_) {
		f.imageAvailable = vk::raii::Semaphore(device_, vk::SemaphoreCreateInfo());
		f.inFlight = vk::raii::Fence(device_, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
	}
}

uint32_t Swapchain::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
	auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
	if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
		minImageCount = surfaceCapabilities.maxImageCount;
	}
	return minImageCount;
}

vk::SurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	assert(!availableFormats.empty());
	const auto formatIt = std::ranges::find_if(
		availableFormats,
		[](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
	return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR Swapchain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
	return std::ranges::any_of(availablePresentModes,
		[](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
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

void Swapchain::createCommandBuffers(vk::raii::CommandPool& commandPool) {

	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
											 .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
	auto cmds = vk::raii::CommandBuffers(device_, allocInfo);

	for (unsigned int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		frames_[i].cmd = std::move(cmds[i]);
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
}

void Swapchain::createDescriptorSets(vk::raii::DescriptorSetLayout& descriptorSetLayout, vk::raii::DescriptorPool& descriptorPool)
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	auto sets = device_.allocateDescriptorSets(allocInfo);
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		frames_[i].descriptorSet = std::move(sets[i]);

		vk::DescriptorBufferInfo bufferInfo{
			.buffer = frames_[i].ubo,
			.offset = 0,
			.range = sizeof(UniformBufferObject)
		};
		vk::DescriptorImageInfo imageInfo{
			.sampler = textureSampler,
			.imageView = textureImageView,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = frames_[i].descriptorSet ,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo = &bufferInfo
			},
			vk::WriteDescriptorSet{
				.dstSet = frames_[i].descriptorSet ,
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

void Swapchain::createVertexBuffer()
{
	vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
	vk::raii::Buffer stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, vertices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer_, vertexBufferMemory_);

	copyBuffer(device_, commandPool_, stagingBuffer, vertexBuffer_, bufferSize, queue_);
}

void Swapchain::createIndexBuffer() {
	vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

	vk::raii::Buffer stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

	void* data = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(data, indices.data(), (size_t)bufferSize);
	stagingBufferMemory.unmapMemory();

	createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer_, indexBufferMemory_);

	copyBuffer(device_, commandPool_, stagingBuffer, indexBuffer_, bufferSize, queue_);
}

void Swapchain::createTextureImage() {
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	vk::DeviceSize imageSize = texWidth * texHeight * 4;

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

	createImage(device_, physicalDevice_, texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

	transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Swapchain::copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
	std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
	vk::BufferImageCopy region{ .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
							   .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
							   .imageOffset = {0, 0, 0}, .imageExtent = {width, height, 1} };
	commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
	endSingleTimeCommands(*commandBuffer);
}

void Swapchain::transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
	auto commandBuffer = beginSingleTimeCommands();

	vk::ImageMemoryBarrier barrier{ .oldLayout = oldLayout, .newLayout = newLayout,
								   .image = image,
								   .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };

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
	textureImageView = createImageView(device_, textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
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


void Swapchain::createUniformBuffers()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		frames_[i].ubo.clear();
		frames_[i].uboMem.clear();
		frames_[i].uboMapped = nullptr;

		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		createBuffer(device_, physicalDevice_, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);

		frames_[i].ubo = std::move(buffer);
		frames_[i].uboMem = std::move(bufferMem);
		frames_[i].uboMapped = frames_[i].uboMem.mapMemory(0, bufferSize);
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