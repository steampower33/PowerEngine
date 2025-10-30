#pragma once

class Texture2D
{
public:
	Texture2D(const std::string texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool);
	Texture2D(const Texture2D& rhs) = delete;
	Texture2D(Texture2D&& rhs) = delete;
	~Texture2D() = default;

	Texture2D& operator=(const Texture2D& rhs) = delete;
	Texture2D& operator=(Texture2D&& rhs) = delete;

	vk::raii::Image texture_image_ = nullptr;
	vk::raii::DeviceMemory texture_image_memory_ = nullptr;
	vk::raii::ImageView texture_image_view_ = nullptr;
	vk::raii::Sampler texture_sampler_ = nullptr;
	vk::Format texture_image_format_ = vk::Format::eUndefined;

	void CreateTextureImage(const std::string& texturePath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool);
	void TransitionImageLayout(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
	void CopyBufferToImage(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);
	std::unique_ptr<vk::raii::CommandBuffer> BeginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& commandPool);
	void EndSingleTimeCommands(vk::raii::Queue& queue, const vk::raii::CommandBuffer& commandBuffer);
	void CreateTextureImageView(vk::raii::Device& device);
	void CreateTextureSampler(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device);

};