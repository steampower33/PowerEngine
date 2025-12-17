#pragma once

struct Camera {

    Camera() = default;
    ~Camera() = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    glm::vec3 position{ 0.0f, 3.76383, 3.10179 };
    float yaw = -90.0f;
    float pitch = -28.52f;
    float fov = 60.0f;
    float camera_move_speed = 2.0f;
    float mouse_move_speed = 0.12f;
    float width = 0.0f;
    float height = 0.0f;

    bool focus_enabled = false;
    bool mouse_move_enabled = false;

    float orbit_yaw_speed = 60.0f;
    float orbit_pitch_speed = 60.0f;
    float zoom_speed = 3.0f;

    glm::vec3 Front() const;
    glm::vec3 Right() const;
    glm::vec3 Up() const;
    glm::mat4 View() const;
    glm::mat4 Proj(float width, float height) const;
    void SetCircularPos(float deltaYawDeg, float deltaPitchDeg);
    void InitOrbitFromCurrentPos();
};
