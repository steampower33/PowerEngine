#include "swapchain.h"
#include "camera.h"

Swapchain::Swapchain(
	GLFWwindow* glfwWindow,
	vk::raii::Device& device,
	vk::raii::PhysicalDevice& physicalDevice,
	vk::SampleCountFlagBits msaaSamples,
	vk::raii::SurfaceKHR& surface
) : glfw_window_(glfwWindow)
{
	CreateSwapchain(physicalDevice, device, surface);
	CreateImageViews(device);
}

Swapchain::~Swapchain()
{
}

void Swapchain::CreateSwapchain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::SurfaceKHR& surface) {
	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	swapchain_extent_ = ChooseSwapExtent(surfaceCapabilities);
	swapchain_surface_format_ = ChooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
	vk::SwapchainCreateInfoKHR swapChainCreateInfo{ .surface = surface,
													.minImageCount = ChooseSwapMinImageCount(surfaceCapabilities),
													.imageFormat = swapchain_surface_format_.format,
													.imageColorSpace = swapchain_surface_format_.colorSpace,
													.imageExtent = swapchain_extent_,
													.imageArrayLayers = 1,
													.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
													.imageSharingMode = vk::SharingMode::eExclusive,
													.preTransform = surfaceCapabilities.currentTransform,
													.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
													.presentMode = ChooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
													.clipped = true };

	swapchain_ = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
	swapchain_images_ = swapchain_.getImages();
	image_count_ = swapchain_images_.size();
}

uint32_t Swapchain::ChooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
	min_image_count_ = std::max(3u, surfaceCapabilities.minImageCount);
	if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < min_image_count_)) {
		min_image_count_ = surfaceCapabilities.maxImageCount;
	}
	return min_image_count_;
}

vk::SurfaceFormatKHR Swapchain::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
	assert(!availableFormats.empty());
	const auto formatIt = std::ranges::find_if(
		availableFormats,
		[](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
	return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::PresentModeKHR Swapchain::ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
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

vk::Extent2D Swapchain::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != 0xFFFFFFFF) {
		return capabilities.currentExtent;
	}
	int width, height;
	glfwGetFramebufferSize(glfw_window_, &width, &height);

	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
}

void Swapchain::CleanupSwapChain() {
	swapchain_image_views_.clear();
	swapchain_ = nullptr;
}

void Swapchain::RecreateSwapChain(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::SurfaceKHR& surface) {
	int width = 0, height = 0;
	glfwGetFramebufferSize(glfw_window_, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(glfw_window_, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();

	CleanupSwapChain();
	CreateSwapchain(physicalDevice, device, surface);
	CreateImageViews(device);
}

void Swapchain::CreateImageViews(vk::raii::Device& device) {
	assert(swapchain_image_views_.empty());

	vk::ImageViewCreateInfo imageViewCreateInfo{
		.viewType = vk::ImageViewType::e2D,
		.format = swapchain_surface_format_.format,
		.components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
	};
	for (auto image : swapchain_images_)
	{
		imageViewCreateInfo.image = image;
		swapchain_image_views_.emplace_back(device, imageViewCreateInfo);
	}
}