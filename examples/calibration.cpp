#include "calibration.h"
#include <opencv\cv.h>
#include <Eigen\Dense>
void CalibrationMethod::addImages(std::vector<uint16_t*> depths,  std::vector<uint8_t*> colors,  int w,  int h,
     std::vector<float> fxs,  std::vector<float> fys,  std::vector<float> pxs,  std::vector<float> pys)
{
    if (n < depths.size()) {
        n = depths.size();
        points.resize(n*n);
        poses.resize(n);
        poses[0] = linalg::translation_matrix<float>(float3(0, 0, 0));
    }

    std::vector<std::vector<cv::KeyPoint>> keypoints(n);
    std::vector<cv::Mat> desc(n);
    std::vector<std::vector<float3>> candpoints(n);

    for (int index = 0; index < n; index++){
        //unpack
        auto color = colors[index];
        auto depth = depths[index];
        auto fx = fxs[index];
        auto fy = fys[index];
        auto px = pxs[index];
        auto py = pys[index];

        cv::Mat colorImage(h, w, CV_8UC3, color);
        cv::Mat depthImage(h, w, CV_16U, depth);
        std::vector< cv::KeyPoint > cv_keypoints1, cv_keypoints2;
        cv::Mat cv_desc1;
        std::vector<float3> ptCld(h*w);
        //generate point cloud
        for (int y = 0; y < h; y++){
            for (int x = 0; x < w; x++) {
                ptCld[y*w + x] = depth[y*w + x] ? float3((x - px)/fx * depth[y*w + x], 
                                                         (y - py)/fy * depth[y*w + x], depth[y*w + x]) : float3(0.0f, 0.0f, 0.0f);
            }
        }
        auto clampi = [](int x, int b) { return std::max(std::min(x, b - 1), 0); };
        auto cv_detect = cv::ORB();

        cv_detect.detect(colorImage, cv_keypoints1);
        // only keep ones with depth
        for (auto kp : cv_keypoints1) {
            auto yi = clampi((int)std::round(kp.pt.y), h);
            auto xi = clampi((int)std::round(kp.pt.x), w);
            if (depth[yi*w + xi] && depth[yi*w + xi] < 2000)
                cv_keypoints2.push_back(kp);
        }
        cv_detect.compute(colorImage, cv_keypoints2, cv_desc1);
        keypoints[index] = cv_keypoints2;
        desc[index] = cv_desc1;

        for (int i = 0; i < cv_keypoints2.size(); i++){
            auto kp = cv_keypoints2[i];
            auto yi = clampi((int)std::round(kp.pt.y), h);
            auto xi = clampi((int)std::round(kp.pt.x), w);
            candpoints[index].push_back(ptCld[yi*w + xi]);
        }
    }
    for (int index = 0; index < n; index++){
        for (int index2 = index + 1; index2 < n; index2++) {
            cv::BFMatcher matcher(cv::NORM_HAMMING, true);
            std::vector< cv::DMatch > cv_matches;

            matcher.match(desc[index], desc[index2], cv_matches);
            for (const auto & match : cv_matches) {
                auto kypt1 = keypoints[index][match.queryIdx];
                auto kypt2 = keypoints[index2][match.trainIdx];
                auto dist = match.distance;
                const int SCORE_DIVIDER = 8;
                if (dist < ((desc[index].cols * 8 + (SCORE_DIVIDER - 1)) / SCORE_DIVIDER)) {
                    points[n*index + index2].push_back(candpoints[index][match.queryIdx]);
                    points[n*index2 + index].push_back(candpoints[index2][match.trainIdx]);
                }
            }
        }
    }
}

float4x4 CalibrationMethod::computePose(int index1, int index2)
{
    auto srcPoints = points[n*index1 + index2];
    auto dstPoints = points[n*index2 + index1];

        Eigen::MatrixXf A( 3,srcPoints.size());
        Eigen::MatrixXf B(3, srcPoints.size());
        for (int i = 0; i < srcPoints.size(); i++) {
            A.col(i) = Eigen::Vector3f(srcPoints[i].x, srcPoints[i].y, srcPoints[i].z);
            B.col(i) = Eigen::Vector3f(dstPoints[i].x, dstPoints[i].y, dstPoints[i].z);
        }
        Eigen::Vector3f mean1 = A.rowwise().mean(), mean2 = B.rowwise().mean();
        Eigen::MatrixXf A_zm = A.colwise() - mean1, B_zm = B.colwise() - mean2, C = A_zm*B_zm.transpose();
        Eigen::JacobiSVD< Eigen::MatrixXf> svd = C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3f UVt = svd.matrixU()*svd.matrixV().transpose();
        Eigen::Vector3f v;
        v << 1, 1, UVt.determinant();
        Eigen::Matrix3f R = svd.matrixU()*v.asDiagonal()*svd.matrixV().transpose();
        Eigen::Vector3f t = mean1 - R*mean2;
        Eigen::Affine3f T(R.adjoint());
        T.translation() = -t;
        linalg::aliases::float3x3 laR(R.data());
        linalg::aliases::float3 laT(t.data());
        auto poseLA2 = linalg::pose_matrix(linalg::rotation_quat(laR), scale*laT);

    return poseLA2;
}

bool CalibrationMethod::solvePose()
{
    using linalg::aliases::int3;
    std::vector<int3> pairs;
    std::vector<int> visited(n, -1);
    std::vector<int> visitedPose(n, -1);

    for (auto & pose : poses) {
        pose =linalg::translation_matrix<float>(float3(0, 0, 0));
    }

    for (int index = 0; index < n; index++){
        for (int index2 = index + 1; index2 < n; index2++) {
            pairs.push_back(int3(index, index2, points[n*index + index2].size()));
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](int3 a, int3 b) { return b.z > a.z;  });
    for (int i = 0; i < pairs.size(); i++) {
        if (pairs[i].z && visited[pairs[i].y] < 0)
            visited[pairs[i].y] = pairs[i].x;
    }
    for (int i = 1; i < visited.size(); i++)
        if (visited[i] < 0)
            return false;
    auto allToZero = false;

    // TODO FIXME revisit this logic
    while (!allToZero) {
        for (int index2 = 1; index2 < visited.size(); index2++) {
            if (visited[index2] == 0 && visitedPose[index2] == -1) {
                auto pairPose = computePose(visited[index2], index2);
                poses[index2] = linalg::mul(pairPose,poses[index2]);
                visitedPose[index2] = 0;
                for (int i = 0; i < visited.size(); i++) {
                    if (visited[i] == index2) {
                        auto pairPoseChild = computePose(index2, i);
                        poses[i] = linalg::mul(poses[index2], pairPoseChild);
                        visited[i] = visited[index2];
                        visitedPose[i] = 0;

                    }
                }
            }
        }

        allToZero = true;
        for (int i = 1; i < visitedPose.size(); i++)
            if (visitedPose[i] != 0)
                allToZero = false;
    }
    return true;
}

float4x4 CalibrationMethod::getPose(int index)
{
    return poses[index];
}