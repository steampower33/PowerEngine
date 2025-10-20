#pragma once

#include "perImage.hpp"
#include "perFrame.hpp"
#include "model.hpp"

struct Camera;


class Swapchain
{
public:
	Swapchain(
		GLFWwindow* glfwWindow,
		vk::raii::Device& device,
		vk::raii::PhysicalDevice& physicalDevice,
		vk::SampleCountFlagBits msaaSamples,
		vk::raii::SurfaceKHR& surface,
		vk::raii::Queue& queue,
		vk::raii::DescriptorPool& descriptorPool,
		vk::raii::CommandPool& commandPool,
		DescriptorSetLayouts& descriptorSetLayouts
	);
	Swapchain(const Swapchain& rhs) = delete;
	Swapchain(Swapchain&& rhs) = delete;
	Swapchain& operator=(const Swapchain& rhs) = delete;
	Swapchain& operator=(Swapchain&& rhs) = delete;
	~Swapchain();

	vk::raii::SwapchainKHR           swapchain_ = nullptr;
	vk::SurfaceFormatKHR             swapchainSurfaceFormat_;
	vk::Extent2D                     swapchainExtent_;
	unsigned int					 minImageCount_ = 0;
	unsigned int					 imageCount_ = 0;

	void draw(bool& framebufferResized, Camera& camera, Pipelines& pipelines, float dt);

private:
	void createSwapchain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface);
	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

	void createPerImages();
	vk::raii::ImageView createSwapchainImageView(vk::Image& image, vk::Format format, vk::raii::Device& device);

	void createColorResources();
	void createDepthResources();

	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();

	void createTimelineSemaphore();
	void createPerFrames(
		vk::raii::DescriptorPool& descriptorPool,
		DescriptorSetLayouts& descriptorSetLayouts);
	void createUBOs();
	void createSSBOs();
	void createCommandBuffers(vk::raii::CommandPool& commandPool);
	void createDescriptorSets(vk::raii::DescriptorPool& descriptorPool,
		DescriptorSetLayouts& descriptorSetLayouts);

	void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
		uint32_t mipLevels);
	void transition_image_layout_custom(
		vk::raii::Image& image,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags aspect_mask
	);
	void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);
	void generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

	void cleanupSwapChain();
	void recreateSwapChain();

private:
	GLFWwindow*					glfwWindow_;
	vk::raii::Device&			device_;
	vk::raii::PhysicalDevice&	physicalDevice_;
	vk::SampleCountFlagBits		msaaSamples_;
	vk::raii::SurfaceKHR&		surface_;
	vk::raii::Queue&			queue_;
	vk::raii::CommandPool&		commandPool_;
	vk::raii::Semaphore			semaphore_ = nullptr;
	uint64_t					timelineValue_ = 0;

	uint32_t currentFrame_ = 0;

	std::vector<PerImage> images_;
	std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> frames_;

	vk::raii::Image colorImage_ = nullptr;
	vk::raii::DeviceMemory colorImageMemory_ = nullptr;
	vk::raii::ImageView colorImageView_ = nullptr;

	vk::raii::Image depthImage_ = nullptr;
	vk::raii::DeviceMemory depthImageMemory_ = nullptr;
	vk::raii::ImageView depthImageView_ = nullptr;

	uint32_t mipLevels;
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::Sampler textureSampler = nullptr;

	std::unique_ptr<Model> model;

	void recordComputeCommandBuffer(Pipelines& pipelines);
	void recordGraphicsCommandBuffer(uint32_t imageIndex, Pipelines& pipelines);
	void transition_image_layout(
		uint32_t imageIndex,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	);
	void updateUniformBuffer(uint32_t currentImage, Camera& camera, float dt);
};