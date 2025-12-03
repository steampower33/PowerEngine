#pragma once

class Context;
class PassManager;

class Texture
{
public:
	Texture(std::string path, std::string filename, Context& context);
	Texture(const Texture& rhs) = delete;
	Texture(Texture&& rhs) = delete;
	~Texture();

	Texture& operator=(const Texture& rhs) = delete;
	Texture& operator=(Texture&& rhs) = delete;

	std::string path_;
	std::string filename_;

	uint32_t mip_levels_;
	bool isCubemap = false;
	uint32_t faces = 1u;

	vk::raii::Image texture_image_ = nullptr;
	vk::raii::DeviceMemory texture_image_memory_ = nullptr;
	vk::raii::ImageView texture_image_view_ = nullptr;
	vk::raii::Sampler texture_sampler_ = nullptr;
	vk::Format texture_image_format_ = vk::Format::eUndefined;

	void CreateTextureImage(const std::string& texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool);
	void TransitionImageLayout(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
	void CopyBufferToImage(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height, ktxTexture* kTexture);
	std::unique_ptr<vk::raii::CommandBuffer> BeginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& commandPool);
	void EndSingleTimeCommands(vk::raii::Queue& queue, const vk::raii::CommandBuffer& commandBuffer);
	void CreateTextureImageView(vk::raii::Device& device);
	void CreateTextureSampler(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device);
	
};