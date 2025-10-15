struct PerFrame {
	vk::raii::Semaphore    imageAvailable{ nullptr };
	vk::raii::Fence        inFlight{ nullptr };
	vk::raii::CommandBuffer cmd{ nullptr };

	vk::raii::Buffer        ubo{ nullptr };
	vk::raii::DeviceMemory  uboMem{ nullptr };
	void* uboMapped = nullptr;

	vk::raii::DescriptorSet descriptorSet{ nullptr };
};