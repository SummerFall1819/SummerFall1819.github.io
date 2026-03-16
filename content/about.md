---
title: "关于"
layout: about
hidemeta: true
---

<style>
.resume-container {
  display: flex;
  gap: 2rem;
  margin-top: 2rem;
}

.resume-left {
  flex: 0 0 280px;
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
}

.resume-right {
  flex: 1;
  min-width: 0;
}

.profile-photo {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  margin: 0 auto 1rem;
  display: block;
  object-fit: cover;
  background: #ddd;
}

.resume-left h3 {
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
  color: #333;
}

.resume-left ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.resume-left li {
  margin-bottom: 0.5rem;
  font-size: 0.95rem;
}

.resume-right h2 {
  border-bottom: 2px solid #333;
  padding-bottom: 0.5rem;
  margin-top: 2rem;
  margin-bottom: 1rem;
}

.resume-right h2:first-child {
  margin-top: 0;
}

.experience-item {
  margin-bottom: 1.5rem;
}

.experience-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
}

.experience-title {
  font-weight: bold;
  font-size: 1.05rem;
}

.experience-date {
  color: #666;
  font-size: 0.9rem;
}

.experience-org {
  color: #555;
  font-style: italic;
  margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
  .resume-container {
    flex-direction: column;
  }

  .resume-left {
    flex: 1;
  }
}
</style>

<div class="resume-container">
  <div class="resume-left">
    <!-- 个人照片 -->
    <img src="/images/profile.jpg" alt="个人照片" class="profile-photo">

    <!-- 基本信息 -->
    <h3>基本信息</h3>
    <ul>
      <li><strong>姓名：</strong>李新元</li>
      <li><strong>出生：</strong>2003年</li>
      <li><strong>身份：</strong>博士生</li>
    </ul>

    <!-- 联系方式 -->
    <h3>联系方式</h3>
    <ul>
      <li>📧 email@example.com</li>
      <li>🔗 GitHub: @username</li>
      <li>🐦 Twitter: @username</li>
    </ul>

    <!-- 研究兴趣 -->
    <h3>研究兴趣</h3>
    <ul>
      <li>Agent 系统</li>
      <li>[研究方向 2]</li>
      <li>[研究方向 3]</li>
    </ul>

    <!-- 技能 -->
    <h3>技能</h3>
    <ul>
      <li>Python / Go / Rust</li>
      <li>Machine Learning</li>
      <li>[其他技能]</li>
    </ul>
  </div>

  <div class="resume-right">
    <!-- 教育背景 -->
    <h2>🎓 教育背景</h2>

    <div class="experience-item">
      <div class="experience-header">
        <span class="experience-title">[学位] - [专业]</span>
        <span class="experience-date">20XX - 至今</span>
      </div>
      <div class="experience-org">[大学名称]</div>
      <p>[简要描述研究方向、导师、主要成果等]</p>
    </div>

    <div class="experience-item">
      <div class="experience-header">
        <span class="experience-title">[学位] - [专业]</span>
        <span class="experience-date">20XX - 20XX</span>
      </div>
      <div class="experience-org">[大学名称]</div>
      <p>[简要描述]</p>
    </div>

    <!-- 研究经历 -->
    <h2>🔬 研究经历</h2>

    <div class="experience-item">
      <div class="experience-header">
        <span class="experience-title">[项目名称]</span>
        <span class="experience-date">20XX - 20XX</span>
      </div>
      <div class="experience-org">[机构/实验室]</div>
      <p>[项目描述、主要工作、成果等]</p>
    </div>

    <div class="experience-item">
      <div class="experience-header">
        <span class="experience-title">[项目名称]</span>
        <span class="experience-date">20XX - 20XX</span>
      </div>
      <div class="experience-org">[机构/实验室]</div>
      <p>[项目描述]</p>
    </div>

    <!-- 工作经历 -->
    <h2>💼 工作经历</h2>

    <div class="experience-item">
      <div class="experience-header">
        <span class="experience-title">[职位]</span>
        <span class="experience-date">20XX - 20XX</span>
      </div>
      <div class="experience-org">[公司名称]</div>
      <p>[工作内容、主要成就等]</p>
    </div>

    <!-- 发表论文 -->
    <h2>📝 发表论文</h2>

    <div class="experience-item">
      <p><strong>[论文标题]</strong><br>
      [作者列表]<br>
      <em>[会议/期刊名称]</em>, 20XX</p>
    </div>

    <!-- 获奖荣誉 -->
    <h2>🏆 获奖荣誉</h2>

    <div class="experience-item">
      <ul>
        <li><strong>[奖项名称]</strong> - [颁发机构], 20XX</li>
        <li><strong>[奖项名称]</strong> - [颁发机构], 20XX</li>
      </ul>
    </div>
  </div>
</div>