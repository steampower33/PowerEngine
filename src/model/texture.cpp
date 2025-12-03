#include "context.h"
#include "pass_manager.h"
#include "vulkan_utils.h"

#include "texture.h"

Texture::Texture(std::string path, std::string filename, Context& context)
{
    path_ = path;
    filename_ = filename;
    std::string fullPath = path_ + "/" + filename_;
    CreateTextureImage(fullPath, context.physical_device_, context.device_, context.queue_, context.command_pool_);
    CreateTextureImageView(context.device_);
    CreateTextureSampler(context.physical_device_, context.device_);
}

Texture::~Texture()
{

}

void Texture::CreateTextureImage(const std::string& texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool) {
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
    ktx_size_t dataSize = ktxTexture_GetDataSize(kTexture);
    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(kTexture);
    mip_levels_ = kTexture->numLevels;

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    vku::CreateBuffer(physicalDevice, device, dataSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, dataSize);
    memcpy(data, ktxTextureData, dataSize);
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

    isCubemap = (kTexture->numFaces == 6);
    faces = isCubemap ? 6u : 1u;
    vku::CreateImage(physicalDevice, device, texWidth, texHeight, kTexture->numLevels, vk::SampleCountFlagBits::e1, textureFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, texture_image_, texture_image_memory_,
        isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags(),
        faces);

    TransitionImageLayout(device, queue, commandPool, texture_image_, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    CopyBufferToImage(device, queue, commandPool, stagingBuffer, texture_image_, texWidth, texHeight, kTexture);
    TransitionImageLayout(device, queue, commandPool, texture_image_, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    ktxTexture_Destroy(kTexture);
}

void Texture::TransitionImageLayout(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    auto commandBuffer = BeginSingleTimeCommands(device, commandPool);

    vk::ImageMemoryBarrier barrier{
    .oldLayout = oldLayout,
    .newLayout = newLayout,
    .image = *image,
    .subresourceRange = {
        vk::ImageAspectFlagBits::eColor,
        0,
        mip_levels_,
        0,
        faces
    }
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

void Texture::CopyBufferToImage(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height, ktxTexture* kTexture)
{
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = BeginSingleTimeCommands(device, commandPool);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(mip_levels_ * faces);

    for (uint32_t level = 0; level < mip_levels_; ++level) {
        for (uint32_t face = 0; face < faces; ++face) {
            ktx_size_t offset = 0;
            // layer = 0, faceSlice = face
            KTX_error_code res =
                ktxTexture_GetImageOffset(kTexture, level, 0, face, &offset);
            if (res != KTX_SUCCESS) {
                throw std::runtime_error("failed to get KTX image offset!");
            }

            vk::BufferImageCopy region{};
            region.bufferOffset = offset;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            region.imageSubresource.mipLevel = level;
            region.imageSubresource.baseArrayLayer = face;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = {
                std::max(1u, width >> level),
                std::max(1u, height >> level),
                1
            };

            regions.push_back(region);
        }
    }

    commandBuffer->copyBufferToImage(*buffer, *image, vk::ImageLayout::eTransferDstOptimal, regions);
    EndSingleTimeCommands(queue, *commandBuffer);
}

void Texture::CreateTextureImageView(vk::raii::Device& device) {
    vk::ImageViewType viewType =
        isCubemap ? vk::ImageViewType::eCube : vk::ImageViewType::e2D;

    texture_image_view_ = vku::CreateImageView(
        device,
        texture_image_,
        texture_image_format_,
        vk::ImageAspectFlagBits::eColor,
        mip_levels_,
        viewType,
        faces // layerCount
    );
}

void Texture::CreateTextureSampler(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device) {
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
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = float(mip_levels_ - 1),
    };
    texture_sampler_ = vk::raii::Sampler(device, samplerInfo);
}

std::unique_ptr<vk::raii::CommandBuffer> Texture::BeginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& commandPool) {
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

void Texture::EndSingleTimeCommands(vk::raii::Queue& queue, const vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}
