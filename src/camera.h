#pragma once

struct Camera {

    Camera() = default;
    ~Camera() = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    glm::vec3 position{ 4.5f, 5.8f, 3.8f };
    float yaw = -140.0f;
    float pitch = -36.0f;
    float fov = 60.0f;    // ¡‹
    float move_speed = 6.0f; // m/s
    float sensitivity = 0.2f;
    float width = 0.0f;
    float height = 0.0f;

    glm::vec3 Front() const;
    glm::vec3 Right() const;
    glm::vec3 Up() const;
    glm::mat4 View() const;
    glm::mat4 Proj(float width, float height) const;
};
