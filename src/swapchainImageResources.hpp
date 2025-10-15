
struct SwapchainImageResources {
public:
    SwapchainImageResources(vk::Image& image, vk::SurfaceFormatKHR& swapchainSurfaceFormat, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::Extent2D& swapchainExtent);

	SwapchainImageResources(const SwapchainImageResources&) = delete;
	SwapchainImageResources& operator=(const SwapchainImageResources&) = delete;

	SwapchainImageResources(SwapchainImageResources&&) noexcept = default;
	SwapchainImageResources& operator=(SwapchainImageResources&&) noexcept = default;

	~SwapchainImageResources();

    vk::Image image_{};
    vk::raii::ImageView imageView_{nullptr};

    vk::raii::Image depthImage_{ nullptr };
	vk::raii::DeviceMemory depthMem_{ nullptr };
	vk::raii::ImageView depthImageView_{ nullptr };
	
	void createSwapchainImageView(vk::raii::Device& device, vk::Format format);
};
