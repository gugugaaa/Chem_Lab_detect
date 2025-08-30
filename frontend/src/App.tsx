import React, { useState, useEffect } from 'react';
import './App.css';

interface SafetyResult {
  hand_safety: {
    left_hand: string;
    right_hand: string;
  };
  coat_safety: string;
  fps: number;
  man_detected: number;
}

interface ScoreResult {
  vessel_info: any;
  gesture_info: any;
  score_result: {
    operation: string;
    score: number;
  };
}

function App() {
  const [safetyImage, setSafetyImage] = useState<string>('');
  const [scoreImage, setScoreImage] = useState<string>('');
  const [safetyResult, setSafetyResult] = useState<SafetyResult | null>(null);
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchSafetyData = async () => {
    try {
      // 获取安全检测结果
      const safetyResponse = await fetch('http://localhost:8000/api/safety_result/');
      const safetyData = await safetyResponse.json();
      setSafetyResult(safetyData);

      // 获取安全检测图片
      const safetyImageResponse = await fetch('http://localhost:8000/api/safety_image/');
      const safetyImageData = await safetyImageResponse.json();
      setSafetyImage(safetyImageData.image);
    } catch (error) {
      console.error('获取安全检测数据失败:', error);
    }
  };

  const fetchScoreData = async () => {
    try {
      // 获取评分结果
      const scoreResponse = await fetch('http://localhost:8000/api/score_result/');
      const scoreData = await scoreResponse.json();
      setScoreResult(scoreData);

      // 获取评分图片
      const scoreImageResponse = await fetch('http://localhost:8000/api/score_image/');
      const scoreImageData = await scoreImageResponse.json();
      setScoreImage(scoreImageData.image);
    } catch (error) {
      console.error('获取评分数据失败:', error);
    }
  };

  const fetchInitialData = async () => {
    await Promise.all([fetchSafetyData(), fetchScoreData()]);
    setLoading(false);
  };

  useEffect(() => {
    fetchInitialData();
    
    // 安全检测每3秒更新一次
    const safetyInterval = setInterval(fetchSafetyData, 3000);
    // 评分检测每秒更新一次
    const scoreInterval = setInterval(fetchScoreData, 1000);
    
    return () => {
      clearInterval(safetyInterval);
      clearInterval(scoreInterval);
    };
  }, []);

  if (loading) {
    return <div className="loading">加载中...</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>化学实验室检测系统</h1>
      </header>

      <div className="main-content">
        {/* 图像显示区域 */}
        <div className="images-container">
          <div className="image-section">
            <h3>安全检测图像</h3>
            {safetyImage ? (
              <img src={safetyImage} alt="安全检测" className="detection-image" />
            ) : (
              <div className="no-image">暂无图像</div>
            )}
          </div>

          <div className="image-section">
            <h3>评分检测图像</h3>
            {scoreImage ? (
              <img src={scoreImage} alt="评分检测" className="detection-image" />
            ) : (
              <div className="no-image">暂无图像</div>
            )}
          </div>
        </div>

        {/* 结果显示区域 */}
        <div className="results-container">
          <div className="result-section">
            <h3>安全检测结果</h3>
            {safetyResult ? (
              <div className="result-content">
                <div className="result-item">
                  <strong>左手状态:</strong> {safetyResult.hand_safety.left_hand}
                </div>
                <div className="result-item">
                  <strong>右手状态:</strong> {safetyResult.hand_safety.right_hand}
                </div>
                <div className="result-item">
                  <strong>实验服状态:</strong> {safetyResult.coat_safety}
                </div>
                <div className="result-item">
                  <strong>检测到人数:</strong> {safetyResult.man_detected}
                </div>
                <div className="result-item">
                  <strong>FPS:</strong> {safetyResult.fps.toFixed(2)}
                </div>
              </div>
            ) : (
              <div className="no-result">暂无安全检测结果</div>
            )}
          </div>

          <div className="result-section">
            <h3>评分结果</h3>
            {scoreResult && scoreResult.score_result ? (
              <div className="result-content">
                <div className="result-item">
                  <strong>操作类型:</strong> {scoreResult.score_result.operation}
                </div>
                <div className="result-item">
                  <strong>评分:</strong> {scoreResult.score_result.score.toFixed(1)}
                </div>
              </div>
            ) : (
              <div className="no-result">暂无评分结果</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
