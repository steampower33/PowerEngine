
class Model
{
public:
	Model(const std::string modelPath, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& queue);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	~Model();

	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;

	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;
	vk::raii::Buffer vertexBuffer_ = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory_ = nullptr;
	vk::raii::Buffer indexBuffer_ = nullptr;
	vk::raii::DeviceMemory indexBufferMemory_ = nullptr;

};