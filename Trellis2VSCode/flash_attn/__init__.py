# dummy package to prevent loading real flash_attn
# when kornia attempts from flash_attn.modules.mha import FlashCrossAttention
# we will let the submodule raise ModuleNotFoundError

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Re-export core functions from the real installed flash_attn package
# (this dummy exists only to silently absorb kornia's import attempts)
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    pass
