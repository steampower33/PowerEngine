
class UboManager {

public:
    UboManager();
    UboManager(const UboManager& rhs) = delete;
    UboManager(UboManager&& rhs) = delete;
    ~UboManager();

    UboManager& operator=(const UboManager& rhs) = delete;
    UboManager& operator=(UboManager&& rhs) = delete;

private:
    std::vector<vk::raii::Buffer> uniformBuffers_;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;


};