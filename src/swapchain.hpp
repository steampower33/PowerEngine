#pragma once

#include "swapchainImageResources.hpp"

class Swapchain
{
public:
	Swapchain(
		GLFWwindow* glfwWindow,
		vk::raii::Device& device,
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::SurfaceKHR& surface,
		uint32_t queueFamilyIndex,
		vk::raii::CommandPool& commandPool,
		vk::raii::DescriptorSetLayout& descriptorSetLayout,
		vk::raii::DescriptorPool& descriptorPool,
		vk::raii::Queue& queue);
	Swapchain(const Swapchain& rhs) = delete;
	Swapchain(Swapchain&& rhs) = delete;
	~Swapchain();

	Swapchain& operator=(const Swapchain& rhs) = delete;
	Swapchain& operator=(Swapchain&& rhs) = delete;

	vk::raii::SwapchainKHR           swapchain_ = nullptr;
	vk::SurfaceFormatKHR             swapchainSurfaceFormat_;
	vk::Extent2D                     swapchainExtent_;

	bool draw(bool& framebufferResized, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout);

private:
	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();

	void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);
	void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

private:
	GLFWwindow* glfwWindow_;
	vk::raii::Device& device_;
	vk::raii::PhysicalDevice& physicalDevice_;
	vk::raii::SurfaceKHR& surface_;
	vk::raii::CommandPool& commandPool_;
	vk::raii::Queue& queue_;

	std::vector<SwapchainImageResources> swapchainImageResources_;

	std::vector<vk::raii::DescriptorSet> descriptorSets_;
	std::vector<vk::raii::CommandBuffer> commandBuffers_;

	std::vector<vk::raii::Buffer> uniformBuffers_;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory_;
	std::vector<void*> uniformBuffersMapped_;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphore_;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphore_;
	std::vector<vk::raii::Fence> inFlightFences_;
	uint32_t semaphoreIndex_ = 0;
	uint32_t currentFrame_ = 0;

	vk::raii::Buffer vertexBuffer_ = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory_ = nullptr;
	vk::raii::Buffer indexBuffer_ = nullptr;
	vk::raii::DeviceMemory indexBufferMemory_ = nullptr;

	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::Sampler textureSampler = nullptr;
	
	void createSwapchain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::SurfaceKHR& surface);
	void createSwapchainImageResources();
	void createCommandBuffers(vk::raii::CommandPool& commandPool);
	void createUniformBuffers();
	void createDescriptorSets(vk::raii::DescriptorSetLayout& descriptorSetLayout, vk::raii::DescriptorPool& descriptorPool);

	void createSyncObjects();

	void cleanupSwapChain();
	void recreateSwapChain();

	void recordCommandBuffer(uint32_t imageIndex, vk::raii::Queue& queue, vk::raii::Pipeline& graphicsPipeline, vk::raii::PipelineLayout& pipelineLayout);

	void transition_image_layout(
		uint32_t imageIndex,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	);

	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

	void updateUniformBuffer(uint32_t currentImage);

	void createVertexBuffer();
	void createIndexBuffer();

	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size, vk::raii::Queue& queue);

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
};