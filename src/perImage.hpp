
struct PerImage {
    vk::Image image_{};
    vk::raii::ImageView imageView_{nullptr};

	vk::raii::Semaphore renderFinished_{ nullptr };

	vk::Fence lastSubmitFence = VK_NULL_HANDLE; 
	
};
