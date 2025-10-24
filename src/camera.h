#pragma once

struct Camera {

    Camera() = default;
    ~Camera() = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    glm::vec3 position{ 4.0f, 4.0f, 4.0f };
    float yaw = 224.3f;   // 오른쪽이 0°, -Z를 보고 시작하려면 -90°
    float pitch = -38.2f;
    float fov = 60.0f;    // 줌
    float move_speed = 4.0f; // m/s
    float sensitivity = 0.1f;

    glm::vec3 Front() const;
    glm::vec3 Right() const;
    glm::vec3 Up() const;
    glm::mat4 View() const;
    glm::mat4 Proj(float width, float height) const;
};
