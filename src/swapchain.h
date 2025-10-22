#pragma once

struct Camera;

class Swapchain
{
public:
	Swapchain(
		GLFWwindow* glfwWindow,
		vk::raii::Device& device,
		vk::raii::PhysicalDevice& physicalDevice,
		vk::SampleCountFlagBits msaaSamples,
		vk::raii::SurfaceKHR& surface
	);
	Swapchain(const Swapchain& rhs) = delete;
	Swapchain(Swapchain&& rhs) = delete;
	Swapchain& operator=(const Swapchain& rhs) = delete;
	Swapchain& operator=(Swapchain&& rhs) = delete;
	~Swapchain();

	GLFWwindow* glfw_window_;

	vk::raii::SwapchainKHR           swapchain_ = nullptr;
	std::vector<vk::Image>           swapchain_images_;
	vk::SurfaceFormatKHR             swapchain_surface_format_;
	vk::Extent2D                     swapchain_extent_;
	std::vector<vk::raii::ImageView> swapchain_image_views_;

	uint16_t min_image_count_ = 0;
	uint16_t image_count_ = 0;

	void RecreateSwapChain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::SurfaceKHR& surface);
private:
	void CreateSwapchain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::SurfaceKHR& surface);
	uint32_t ChooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	void CreateImageViews(vk::raii::Device& device);

	void CleanupSwapChain();

};