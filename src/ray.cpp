#include "model.h"
#include <limits> // std::numeric_limits 사용

#include "ray.h"

Ray::Ray(const glm::vec3& origin, const glm::vec3& direction)
    : origin(origin), direction(glm::normalize(direction)) // 생성자에서 바로 정규화
{
}

// 이 함수가 바로 네가 올린 RaySphereIntersect의 GLM 버전이야.
bool Ray::Intersects(const Model& model, float& dist) const
{
    // 광선의 시작점에서 구의 중심을 향하는 벡터
    glm::vec3 oc = origin - model.position_;

    // 2차 방정식의 계수 계산
    // a = dot(D, D) : 광선 방향 벡터의 내적. 정규화되어 있으므로 항상 1.0
    const float a = 1.0f;

    // b = 2 * dot(OC, D)
    const float b = 2.0f * glm::dot(oc, direction);

    // c = dot(OC, OC) - r^2
    const float c = glm::dot(oc, oc) - model.radius_ * model.radius_;

    // 판별식 (b^2 - 4ac)
    const float discriminant = b * b - 4.0f * a * c;

    // 판별식이 0보다 작으면 실근이 없으므로, 충돌하지 않음
    if (discriminant < 0.0f) {
        return false;
    }

    // 두 개의 잠재적인 충돌 거리(t)를 계산
    const float sqrt_discriminant = sqrt(discriminant);
    const float t0 = (-b - sqrt_discriminant) / (2.0f * a);
    const float t1 = (-b + sqrt_discriminant) / (2.0f * a);

    // 광선은 한 방향으로만 나아가므로, t값은 양수여야 함.
    // 두 개의 양수 t값 중 더 작은 값(더 가까운 교차점)을 선택한다.

    float closest_t = std::numeric_limits<float>::max();
    bool hit = false;

    if (t0 > 0.0001f) { // 아주 작은 오차 허용
        closest_t = t0;
        hit = true;
    }

    if (t1 > 0.0001f && t1 < closest_t) {
        closest_t = t1;
        hit = true;
    }

    if (hit) {
        dist = closest_t;
        return true;
    }

    return false;
}