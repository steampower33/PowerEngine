#include "camera.h"

glm::vec3 Camera::Front() const {
    float cy = cos(glm::radians(yaw));
    float sy = sin(glm::radians(yaw));
    float cp = cos(glm::radians(pitch));
    float sp = sin(glm::radians(pitch));

    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
}
glm::vec3 Camera::Right() const {
    return glm::normalize(glm::cross(Front(), glm::vec3(0, 1, 0)));
}
glm::vec3 Camera::Up() const {
    return glm::normalize(glm::cross(Right(), Front()));
}

glm::mat4 Camera::View() const {
    //std::cout << position.x << ", " << position.y << ", " << position.z << std::endl;
    //std::cout << yaw << std::endl;
    //std::cout << pitch << std::endl;
    if (focus_enabled)
        return glm::lookAt(position, glm::vec3(0.0f), glm::vec3(0, 1, 0));
    else
        return glm::lookAt(position, position + Front(), glm::vec3(0, 1, 0));
}
glm::mat4 Camera::Proj(float width, float height) const {
    glm::mat4 p = glm::perspective(glm::radians(fov), width / height, 0.1f, 1000.0f);
    p[1][1] *= -1;
    return p;
}

void Camera::SetCircularPos(float deltaYawDeg, float deltaPitchDeg)
{
    yaw += deltaYawDeg;
    pitch += deltaPitchDeg;
    pitch = glm::clamp(pitch, -89.0f, 89.0f);

    float radius = glm::length(position);
    if (radius < 0.001f) radius = 0.001f;

    glm::vec3 dir = Front();

    position = -dir * radius;
}

void Camera::InitOrbitFromCurrentPos()
{
    float radius = glm::length(position);
    if (radius < 0.0001f) radius = 0.0001f;

    glm::vec3 dir = glm::normalize(position);

    pitch = glm::degrees(asinf(dir.y));

    yaw = glm::degrees(atan2f(dir.z, dir.x));
}