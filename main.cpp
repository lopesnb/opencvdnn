#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // 1. 学習済みモデルの読み込み（知能をロード）
   // auto net = cv::dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
    auto net = cv::dnn::readNetFromONNX("best.onnx");
//追加開始-------------------------------------------------------
//CUDAバックエンド（GPU）を使用する設定
//    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
//追加終了-------------------------------------------------------

    cv::VideoCapture cap(0); // カメラ開始
    cv::Mat frame;

    while (cap.read(frame)) {
        // 2. 画像をAIが読める形式（Blob）に変換
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0));
        net.setInput(blob);

        // 3. 推論（Playgroundでいう「Playボタン」実行）
        cv::Mat detection = net.forward();

        // 4. 結果の表示（確率が高いものに枠を描く）
        // (ここに数行、枠を描画する処理が入ります)
// --- ここから追加：検出結果を解析して枠を描く ---
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > 0.5) { // 50%以上の確信度があれば表示
                // 座標を計算（0.0~1.0の割合で返ってくるので、画像のピクセルサイズに直す）
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                // 枠を描画（緑色、太さ2）
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                
                // 確率も文字で出してみる
                std::string label = cv::format("Log: %.2f", confidence);
                cv::putText(frame, label, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            }
        }
        // --- ここまで追加 ---        
        cv::imshow("AI Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}