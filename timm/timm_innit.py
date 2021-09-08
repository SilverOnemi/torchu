from .version import __version__
from .models import create_model, list_models, is_model, list_modules, model_entrypoint, \
    is_scriptable, is_exportable, set_scriptable, set_exportable, has_model_default_key, is_model_default_key, \
    get_model_default_value, is_model_pretrained

from .loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from .optim import create_optimizer_v2, optimizer_kwargs
from .scheduler import create_scheduler