#include "IrisLandmark.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>

#define SHOW_FPS (1)

#if SHOW_FPS
#include <chrono>
#endif

#include <ApplicationServices/ApplicationServices.h>
// #include <unistd.h>

int main(int argc, char *argv[])
{
    CGEventRef eventGet = CGEventCreate(NULL);
    CGPoint cursor = CGEventGetLocation(eventGet);
    CFRelease(eventGet);
    std::cout << cursor.x << ", " << cursor.y << "\n\n";

    CGPoint location = CGPointMake(500, 200);
    CGEventRef eventSet = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, location, kCGMouseButtonLeft);
    CGEventSetType(eventSet, kCGEventMouseMoved);
    CGEventPost(kCGHIDEventTap, eventSet);
    CFRelease(eventSet);

    my::IrisLandmark irisLandmarker("./models");
    cv::VideoCapture cap(0);

    bool success = cap.isOpened();
    if (success == false)
    {
        std::cerr << "Cannot open the camera." << std::endl;
        return 1;
    }

#if SHOW_FPS
    float sum = 0;
    int count = 0;
#endif

    while (success)
    {
        cv::Mat rframe, frame;
        success = cap.read(rframe); // read a new frame from video

        // cv::resize(rframe, rframe, cv::Size(320, 240));
        // rframe = rframe(cv::Range(20, 200), cv::Range(20, 200));

        if (success == false)
            break;

        cv::flip(rframe, rframe, 1);

#if SHOW_FPS
        auto start = std::chrono::high_resolution_clock::now();
#endif

        irisLandmarker.loadImageToInput(rframe);
        irisLandmarker.runInference();

        for (auto landmark : irisLandmarker.getAllFaceLandmarks())
        {
            cv::circle(rframe, landmark, 2, cv::Scalar(0, 255, 0), -1);
        }

        for (auto landmark : irisLandmarker.getAllEyeLandmarks(true, true))
        {
            cv::circle(rframe, landmark, 2, cv::Scalar(0, 0, 255), -1);
        }

        for (auto landmark : irisLandmarker.getAllEyeLandmarks(false, true))
        {
            cv::circle(rframe, landmark, 2, cv::Scalar(0, 0, 255), -1);
        }

#if SHOW_FPS
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float inferenceTime = duration.count() / 1e3;
        sum += inferenceTime;
        count += 1;
        int fps = (int)1e3 / inferenceTime;

        cv::putText(rframe, std::to_string(fps), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 196, 255), 2);
#endif

        cv::imshow("Face detector", rframe);

        if (cv::waitKey(10) == 27)
            break;
    }

#if SHOW_FPS
    std::cout << "Average inference time: " << sum / count << "ms " << std::endl;
#endif

    cap.release();
    cv::destroyAllWindows();
    return 0;
}