#include "texture_2d.hpp"
#include "vulkan_utils.h"

Texture2D::Texture2D(const std::string texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool)
{
    CreateTextureImage(texturePath, physicalDevice, device, queue, commandPool);
    CreateTextureImageView(device);
    CreateTextureSampler(physicalDevice, device);
}

void Texture2D::CreateTextureImage(const std::string& texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool) {
    // Load KTX2 texture instead of using stb_image
    ktxTexture* kTexture;
    KTX_error_code result = ktxTexture_CreateFromNamedFile(
        texturePath.c_str(),
        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &kTexture);

    if (result != KTX_SUCCESS) {
        throw std::runtime_error("failed to load ktx texture image!");
    }

    // Get texture dimensions and data
    uint32_t texWidth = kTexture->baseWidth;
    uint32_t texHeight = kTexture->baseHeight;
    ktx_size_t imageSize = ktxTexture_GetImageSize(kTexture, 0);
    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(kTexture);

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    vku::CreateBuffer(physicalDevice, device, imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, ktxTextureData, imageSize);
    stagingBufferMemory.unmapMemory();

    // Determine the Vulkan format from KTX format
    vk::Format textureFormat;

    // Check if the KTX texture has a format
    if (kTexture->classId == ktxTexture2_c) {
        // For KTX2 files, we can get the format directly
        auto* ktx2 = reinterpret_cast<ktxTexture2*>(kTexture);
        textureFormat = static_cast<vk::Format>(ktx2->vkFormat);
        if (textureFormat == vk::Format::eUndefined) {
            // If the format is undefined, fall back to a reasonable default
            textureFormat = vk::Format::eR8G8B8A8Unorm;
        }
    }
    else {
        // For KTX1 files or if we can't determine the format, use a reasonable default
        textureFormat = vk::Format::eR8G8B8A8Unorm;
    }

    texture_image_format_ = textureFormat;

    vku::CreateImage(physicalDevice, device, texWidth, texHeight, kTexture->numLevels, vk::SampleCountFlagBits::e1, textureFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, texture_image_, texture_image_memory_);

    TransitionImageLayout(device, queue, commandPool, texture_image_, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    CopyBufferToImage(device, queue, commandPool, stagingBuffer, texture_image_, texWidth, texHeight);
    TransitionImageLayout(device, queue, commandPool, texture_image_, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    ktxTexture_Destroy(kTexture);
}

void Texture2D::TransitionImageLayout(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    auto commandBuffer = BeginSingleTimeCommands(device, commandPool);

    vk::ImageMemoryBarrier barrier{
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .image = *image,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
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
    EndSingleTimeCommands(queue, *commandBuffer);
}

void Texture2D::CopyBufferToImage(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = BeginSingleTimeCommands(device, commandPool);
    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1}
    };
    commandBuffer->copyBufferToImage(*buffer, *image, vk::ImageLayout::eTransferDstOptimal, { region });
    EndSingleTimeCommands(queue, *commandBuffer);
}

void Texture2D::CreateTextureImageView(vk::raii::Device& device) {
    texture_image_view_ = vku::CreateImageView(device, texture_image_, texture_image_format_, vk::ImageAspectFlagBits::eColor, 1);
}

void Texture2D::CreateTextureSampler(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device) {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
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
        .compareOp = vk::CompareOp::eAlways
    };
    texture_sampler_ = vk::raii::Sampler(device, samplerInfo);
}

std::unique_ptr<vk::raii::CommandBuffer> Texture2D::BeginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& commandPool) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

    vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };
    commandBuffer->begin(beginInfo);

    return commandBuffer;
}

void Texture2D::EndSingleTimeCommands(vk::raii::Queue& queue, const vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}
