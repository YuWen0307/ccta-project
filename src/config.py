"""配置加载模块"""
"""
1._dict_to_obj:递归把YAML字典转成可点访问的对象,兼容整数键/列表嵌套
2.Config.__init__:自动定位默认配置路径、安全读取YAML、挂载所有配置项为实例属性
3.Config._resolve_paths:把相对路径统一锚定到项目根、自动创建所有工作目录、兼容绝对路径
4.load_config:对外提供极简统一的加载入口，隐藏内部实现、预留扩展能力
5.__main__自测块:一键验证全流程,快速定位配置错误
"""
import os
import yaml
from types import SimpleNamespace # 标准库里现成的「属性容器」
from pathlib import Path

# 把 YAML 读出来的普通 dict 递归转换成‌可以用「点」访问属性的对象‌
def _dict_to_obj(d):
    """递归将dict转为支持属性访问的SimpleNamespace
    对含非字符串key的dict,保持为dict但递归处理value;
    若key全是字符串,则转SimpleNamespace。
    """
    if isinstance(d, dict):
        if all(isinstance(k, str) for k in d.keys()):
            return SimpleNamespace(**{k: _dict_to_obj(v) for k, v in d.items()})
        else:
            # 混合/整数key：保留dict，value继续递归
            return {k: _dict_to_obj(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_dict_to_obj(x) for x in d]
    return d


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
            # Path(__file__) → 源码中 src/config.py 的绝对路径,两次 .parent 上溯两级：src/ → ‌项目根目录
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) 
            # 只解析字典、列表、标量等基础类型，不会执行YAML中嵌入的任意Python代码

        self.project_root = Path(config_path).parent.parent
        # 固定project_root项目根路径，作为后续所有相对路径的拼接基准
        for k, v in cfg.items():
            setattr(self, k, _dict_to_obj(v))
        # 遍历YAML的所有顶级键（如data、features、output、preprocessing），通过setattr动态挂载为Config实例的属性
        # 例如，cfg.data.raw_dir → self.data.raw_dir

        # 解析为绝对路径
        self._resolve_paths()
    
    # 把配置里写的相对路径，转换成真正可用的系统绝对路径
    def _resolve_paths(self):
        self.data.raw_dir = self.project_root / self.data.raw_dir
        self.data.processed_dir = self.project_root / self.data.processed_dir
        self.output.reports_dir = self.project_root / self.output.reports_dir
        self.output.per_patient_dir = self.project_root / self.output.per_patient_dir
        # 这里用/运算符拼接路径：‌如果右边的路径本身是绝对路径，会自动忽略左边的项目根，直接返回绝对路径‌
        
        # 确保目录存在 把所有后续要写文件的目录提前建好
        for d in [self.data.raw_dir, self.data.processed_dir,
                  self.output.reports_dir,
                  self.output.per_patient_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # parents=True‌：递归创建所有父目录 exist_ok=True‌：目录已经存在时不抛错

# load_config 工厂函数：极简统一的对外入口
def load_config(config_path=None):
    """加载配置"""
    return Config(config_path)

# 最常用的模块自测写法
if __name__ == "__main__":
    cfg = load_config()
    print("配置加载成功")
    print(f"标签映射: {cfg.labels}")
