#include "camera.hpp"

glm::vec3 Camera::front() const {
    float cy = cos(glm::radians(yaw));
    float sy = sin(glm::radians(yaw));
    float cp = cos(glm::radians(pitch));
    float sp = sin(glm::radians(pitch));
    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
}
glm::vec3 Camera::right() const {
    return glm::normalize(glm::cross(front(), glm::vec3(0, 1, 0)));
}
glm::vec3 Camera::up() const {
    return glm::normalize(glm::cross(right(), front()));
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + front(), glm::vec3(0, 1, 0));
}
glm::mat4 Camera::proj(float width, float height) const {
    glm::mat4 p = glm::perspective(glm::radians(fov), width / height, 0.1f, 1000.0f);
    // GLM은 OpenGL 기준으로 Y를 뒤집지 않지만, Vulkan 스크린 좌표계에 맞추려면 보통 아래 한 줄 추가
    p[1][1] *= -1; // Vulkan NDC 보정 (글꼴/좌표계 설정에 따라 제거 가능)
    return p;
}