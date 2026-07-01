import torch
import inspect
import sys
import os

# 忽略那些烦人的 UserWarning
import warnings

warnings.filterwarnings("ignore")


def check_model_build():
    print("🚀 开始模型参数匹配自检 (Final Version)...")

    try:
        from nnunetv2.nets.SwinUMamba import SwinUMamba
        print("✅ 成功导入 SwinUMamba 类")

        # 1. 打印签名 (再次确认)
        sig = inspect.signature(SwinUMamba.__init__)
        print(f"📋 官方定义的参数列表: {sig}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2. 使用正确的参数构建模型
        # === 关键修正 ===
        # 1. in_channels -> in_chans
        # 2. num_classes -> out_chans
        # 3. 删除 depths 和 lengths (这是导致刚才报错的原因，因为它们不在官方定义里)
        print("\n🔄 尝试使用 [Correct Signature] 参数构建模型...")

        try:
            model = SwinUMamba(
                in_chans=3,  # 输入通道 (RGB=3)
                out_chans=2,  # 输出类别 (背景+息肉=2)
                deep_supervision=False  # 关闭深监督以便简化测试
            ).to(device)
            print("✅ 模型构建成功！")
        except TypeError as e:
            print(f"❌ 构建失败: {e}")
            print("   -> 请检查参数是否与上方'官方定义'列表一致")
            return

        # 3. 前向传播测试
        print("🔄 正在测试前向传播 (Forward Pass)...")
        # 模拟一张 352x352 的图片
        dummy_input = torch.randn(1, 3, 352, 352).to(device)

        # 运行模型
        output = model(dummy_input)

        # 处理输出 (Swin-UMamba 有时返回 list, 有时返回 tensor)
        if isinstance(output, (list, tuple)):
            out_shape = output[0].shape
            print("ℹ️ 模型返回的是 List/Tuple (Deep Supervision 格式)")
        else:
            out_shape = output.shape
            print("ℹ️ 模型返回的是 Tensor")

        print(f"\n🎉🎉🎉 完美通过！")
        print(f"   -> 输入: {dummy_input.shape}")
        print(f"   -> 输出: {out_shape}")
        print("   -> 结论: 环境完全就绪，可以运行 Step 3 训练脚本了！")

    except ImportError:
        print("❌ 找不到 swin_umamba/nnunetv2，请确认 pip install -e . 已执行")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_model_build()