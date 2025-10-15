
struct PerImage {
    vk::Image image_{};
    vk::raii::ImageView imageView_{nullptr};

    vk::raii::Image depthImage_{ nullptr };
	vk::raii::DeviceMemory depthMem_{ nullptr };
	vk::raii::ImageView depthImageView_{ nullptr };

	vk::raii::Semaphore renderFinished_{ nullptr };

	vk::Fence lastSubmitFence = VK_NULL_HANDLE; 
	
};
