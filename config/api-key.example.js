/**
 * ============================================================================
 * API 密钥配置模板
 * ============================================================================
 * 
 * 设置步骤：
 * 
 *   第一步：复制此文件
 *     将 "api-key.example.js" 复制为 "api-key.js"
 *     命令：copy api-key.example.js api-key.js
 * 
 *   第二步：获取 OpenAI API 密钥
 *     a. 访问 https://platform.openai.com/account/api-keys
 *     b. 登录或注册账号
 *     c. 点击 "Create new secret key"（创建新密钥）
 *     d. 复制密钥（以 "sk-" 开头）
 * 
 *   第三步：粘贴你的 API 密钥
 *     将下方的 'sk-your-api-key-here' 替换为你的实际密钥
 * 
 *   第四步：设置使用限额（推荐）
 *     a. 访问 https://platform.openai.com/account/limits
 *     b. 设置月度使用限额，防止意外扣费
 * 
 * 安全提示：
 *   - api-key.js 已加入 .gitignore，不会上传到 GitHub
 *   - 切勿与任何人分享你的 API 密钥
 *   - 如果密钥泄露，请立即在 OpenAI 后台重新生成
 * 
 * 费用参考：
 *   - GPT-4o-mini：约 $0.15/百万输入token，$0.60/百万输出token
 *   - text-embedding-3-small：约 $0.02/百万token
 *   - 生成 200 道题目大约花费 $0.10-0.30
 * 
 * ============================================================================
 */

// 将 'sk-your-api-key-here' 替换为你的实际 OpenAI API 密钥
const OPENAI_API_KEY = 'sk-your-api-key-here';

// 自动将 API 密钥存储到 localStorage
// 这样 medical-quiz.html 就能读取并使用它
if (typeof OPENAI_API_KEY !== 'undefined' && OPENAI_API_KEY && OPENAI_API_KEY.startsWith('sk-')) {
    localStorage.setItem('openai_api_key', OPENAI_API_KEY);
    console.log('✅ API Key loaded from api-key.js');
}
