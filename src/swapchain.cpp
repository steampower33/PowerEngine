#include "swapchain.hpp"

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
    createSwapChain(physicalDevice, surface);
    createImageViews();

    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    createVertexBuffer();
    createIndexBuffer();

    createUniformBuffers();
    createDescriptorSets(descriptorSetLayout, descriptorPool);
    createCommandBuffers(commandPool);
    createSyncObjects();
}

Swapchain::~Swapchain()
{
}

bool Swapchain::draw(bool& framebufferResized, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout)
{
    while (vk::Result::eTimeout == device_.waitForFences(*inFlightFences_[currentFrame_], vk::True, UINT64_MAX))
        ;
    uint32_t imageIndex = 0;
    try {
        auto [res, idx] = swapChain_.acquireNextImage(
            UINT64_MAX, *presentCompleteSemaphore_[semaphoreIndex_], nullptr);
        imageIndex = idx;
        if (res == vk::Result::eSuboptimalKHR) {
            framebufferResized = false;
            recreateSwapChain();
            return true;
        }
    }
    catch (const vk::OutOfDateKHRError&) {
        framebufferResized = false;
        recreateSwapChain();
        return true;
    }

    updateUniformBuffer(currentFrame_);

    device_.resetFences(*inFlightFences_[currentFrame_]);
    commandBuffers_[currentFrame_].reset();
    recordCommandBuffer(imageIndex, queue, graphicsPipeline, pipelineLayout);

    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*presentCompleteSemaphore_[semaphoreIndex_],
                        .pWaitDstStageMask = &waitDestinationStageMask, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers_[currentFrame_],
                        .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore_[imageIndex] };
    queue.submit(submitInfo, *inFlightFences_[currentFrame_]);

    const vk::PresentInfoKHR presentInfoKHR{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore_[imageIndex],
                                            .swapchainCount = 1, .pSwapchains = &*swapChain_, .pImageIndices = &imageIndex };
    try {
        queue.presentKHR(presentInfoKHR);   // 실패 시 예외 발생 (리턴값 없음)
    }
    catch (const vk::OutOfDateKHRError&) {
        framebufferResized = false;
        recreateSwapChain();
        return true;
    }
    catch (const vk::SystemError& e) {
        // 일부 버전에선 SuboptimalKHR가 SystemError로 던져진다
        if (e.code() == vk::make_error_code(vk::Result::eSuboptimalKHR)) {
            framebufferResized = false;
            recreateSwapChain();
            return true;
        }
        throw; // 다른 에러는 그대로 위로
    }
    semaphoreIndex_ = (semaphoreIndex_ + 1) % presentCompleteSemaphore_.size();
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;

    return false;
}

void Swapchain::recordCommandBuffer(uint32_t imageIndex, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout) {
    commandBuffers_[currentFrame_].begin({});
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
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo attachmentInfo = {
        .imageView = swapChainImageViews_[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent_ },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo
    };
    commandBuffers_[currentFrame_].beginRendering(renderingInfo);
    commandBuffers_[currentFrame_].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    commandBuffers_[currentFrame_].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent_.width), static_cast<float>(swapChainExtent_.height), 0.0f, 1.0f));
    commandBuffers_[currentFrame_].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent_));
    commandBuffers_[currentFrame_].bindVertexBuffers(0, *vertexBuffer_, { 0 });
    commandBuffers_[currentFrame_].bindIndexBuffer(indexBuffer_, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
    commandBuffers_[currentFrame_].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets_[currentFrame_], nullptr);
    commandBuffers_[currentFrame_].drawIndexed(indices.size(), 1, 0, 0, 0);
    commandBuffers_[currentFrame_].endRendering();
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
    commandBuffers_[currentFrame_].end();
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
        .image = swapChainImages_[imageIndex],
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
    commandBuffers_[currentFrame_].pipelineBarrier2(dependency_info);
}

void Swapchain::updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent_.width) / static_cast<float>(swapChainExtent_.height), 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped_[currentImage], &ubo, sizeof(ubo));
}

void Swapchain::createSwapChain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface) {
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    swapChainExtent_ = chooseSwapExtent(surfaceCapabilities);
    swapChainSurfaceFormat_ = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
    vk::SwapchainCreateInfoKHR swapChainCreateInfo{ .surface = surface,
                                                    .minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
                                                    .imageFormat = swapChainSurfaceFormat_.format,
                                                    .imageColorSpace = swapChainSurfaceFormat_.colorSpace,
                                                    .imageExtent = swapChainExtent_,
                                                    .imageArrayLayers = 1,
                                                    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                                    .imageSharingMode = vk::SharingMode::eExclusive,
                                                    .preTransform = surfaceCapabilities.currentTransform,
                                                    .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                    .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
                                                    .clipped = true };

    swapChain_ = vk::raii::SwapchainKHR(device_, swapChainCreateInfo);
    swapChainImages_ = swapChain_.getImages();
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


void Swapchain::createImageViews() {
    assert(swapChainImageViews_.empty());

    vk::ImageViewCreateInfo imageViewCreateInfo{ .viewType = vk::ImageViewType::e2D, .format = swapChainSurfaceFormat_.format,
      .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
    for (auto image : swapChainImages_)
    {
        imageViewCreateInfo.image = image;
        swapChainImageViews_.emplace_back(device_, imageViewCreateInfo);
    }
}

void Swapchain::cleanupSwapChain() {
    swapChainImageViews_.clear();
    swapChain_ = nullptr;
}

void Swapchain::createSyncObjects() {
    presentCompleteSemaphore_.clear();
    renderFinishedSemaphore_.clear();
    inFlightFences_.clear();

    for (size_t i = 0; i < swapChainImages_.size(); i++) {
        presentCompleteSemaphore_.emplace_back(device_, vk::SemaphoreCreateInfo());
        renderFinishedSemaphore_.emplace_back(device_, vk::SemaphoreCreateInfo());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences_.emplace_back(device_, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
    }
}

void Swapchain::createCommandBuffers(vk::raii::CommandPool& commandPool) {
    commandBuffers_.clear();
    vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary,
                                             .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
    commandBuffers_ = vk::raii::CommandBuffers(device_, allocInfo);
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
    createSwapChain(physicalDevice_, surface_);
    createImageViews();
}

void Swapchain::createUniformBuffers()
{
    uniformBuffers_.clear();
    uniformBuffersMemory_.clear();
    uniformBuffersMapped_.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers_.emplace_back(std::move(buffer));
        uniformBuffersMemory_.emplace_back(std::move(bufferMem));
        uniformBuffersMapped_.emplace_back(uniformBuffersMemory_[i].mapMemory(0, bufferSize));
    }
}

void Swapchain::createDescriptorSets(vk::raii::DescriptorSetLayout& descriptorSetLayout, vk::raii::DescriptorPool& descriptorPool)
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    descriptorSets_.clear();
    descriptorSets_ = device_.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = uniformBuffers_[i],
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
                .dstSet = descriptorSets_[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
            },
            vk::WriteDescriptorSet{
                .dstSet = descriptorSets_[i],
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
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer_, vertexBufferMemory_);

    copyBuffer(stagingBuffer, vertexBuffer_, bufferSize, queue_);
}

void Swapchain::createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer_, indexBufferMemory_);

    copyBuffer(stagingBuffer, indexBuffer_, bufferSize, queue_);
}

void Swapchain::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
    buffer = vk::raii::Buffer(device_, bufferInfo);
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
    bufferMemory = vk::raii::DeviceMemory(device_, allocInfo);
    buffer.bindMemory(bufferMemory, 0);
}

uint32_t Swapchain::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice_.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void Swapchain::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size, vk::raii::Queue& queue) {
    vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool_, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    vk::raii::CommandBuffer commandCopyBuffer = std::move(device_.allocateCommandBuffers(allocInfo).front());
    commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
    commandCopyBuffer.end();
    queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
    queue.waitIdle();
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
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Swapchain::createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
    vk::ImageCreateInfo imageInfo{ .imageType = vk::ImageType::e2D, .format = format,
                                  .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                  .samples = vk::SampleCountFlagBits::e1, .tiling = tiling,
                                  .usage = usage, .sharingMode = vk::SharingMode::eExclusive };

    image = vk::raii::Image(device_, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size,
                                     .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
    imageMemory = vk::raii::DeviceMemory(device_, allocInfo);
    image.bindMemory(imageMemory, 0);
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
    textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
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

vk::raii::ImageView Swapchain::createImageView(vk::raii::Image& image, vk::Format format) {
    vk::ImageViewCreateInfo viewInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
    return vk::raii::ImageView(device_, viewInfo);
}