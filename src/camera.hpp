
struct Camera {

    Camera();
    Camera(const Camera& rhs) = delete;
    Camera(Camera&& rhs) = delete;
    ~Camera();

    Camera& operator=(const Camera& rhs) = delete;
    Camera& operator=(Camera&& rhs) = delete;

    glm::vec3 position{ 0.0f, 0.0f, 3.0f };
    float yaw = -90.0f;   // 오른쪽이 0°, -Z를 보고 시작하려면 -90°
    float pitch = 0.0f;
    float fov = 60.0f;    // 줌
    float moveSpeed = 4.0f; // m/s
    float sensitivity = 0.1f;

    glm::vec3 front() const;
    glm::vec3 right() const;
    glm::vec3 up() const;
    glm::mat4 view() const;
    glm::mat4 proj(float width, float height) const;
};
