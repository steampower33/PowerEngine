#pragma once

class Model;

struct BoundingSphere
{
    glm::vec3 center{ 0.0f };
    float radius = 0.0f;
};

class Ray
{
public:
    // 생성자: 방향 벡터는 생성 시 자동으로 정규화(normalize)된다.
    Ray(const glm::vec3& origin, const glm::vec3& direction);

    // 광선과 구의 충돌을 검사하는 멤버 함수
    // dist: [출력용] 충돌 시, 광선 시작점부터의 거리를 담을 변수
    // 반환값: 충돌 여부
    bool Intersects(const Model& model, float& dist) const;

public:
    glm::vec3 origin;
    glm::vec3 direction; // 항상 정규화된 상태
};