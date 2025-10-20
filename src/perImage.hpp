
struct PerImage {
    vk::Image image_{};
    vk::raii::ImageView imageView_{nullptr};
};
