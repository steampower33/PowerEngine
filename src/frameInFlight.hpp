
class FrameInflight {
public:
    FrameInflight();
    FrameInflight(const FrameInflight& rhs) = delete;
    FrameInflight(FrameInflight&& rhs) = delete;
    ~FrameInflight();

private:
    vk::raii::CommandBuffer cmd_{ nullptr };

    vk::raii::Buffer        ubo_{ nullptr };
    vk::raii::DeviceMemory  uboMem_{ nullptr };
    void* uboMapped_ = nullptr;
};