from __future__ import unicode_literals

from builtins import str

from snips_nlu.builtin_entities import is_builtin_entity
from snips_nlu.constants import (RES_SLOTS, RES_ENTITY, RES_INTENT)
from snips_nlu.nlu_engine.nlu_engine import SnipsNLUEngine
from snips_nlu.nlu_engine.utils import resolve_slots
from snips_nlu.pipeline.configs import DENLUEngineConfig
from snips_nlu.result import empty_result, is_empty, is_empty_list, parsing_result
from snips_nlu.utils import NotTrained


class DESnipsNLUEngine(SnipsNLUEngine):
    unit_name = "de_nlu_engine"
    config_type = DENLUEngineConfig

    def parse(self, text, intents=None):
        """Performs intent parsing on the provided *text* by calling its intent
        parsers successively

        Args:
            text (str): Input
            intents (str or list of str): If provided, reduces the scope of
                intent parsing to the provided list of intents

        Returns:
            dict: The most likely intent along with the extracted slots. See
            :func:`.parsing_result` for the output format.

        Raises:
            NotTrained: When the nlu engine is not fitted
            TypeError: When input type is not unicode
        """

        if not isinstance(text, str):
            raise TypeError("Expected unicode but received: %s" % type(text))

        if not self.fitted:
            raise NotTrained("SnipsNLUEngine must be fitted")

        if isinstance(intents, str):
            intents = [intents]

        language = self._dataset_metadata["language_code"]
        entities = self._dataset_metadata["entities"]

        for parser in self.intent_parsers:
            res = parser.parse(text, intents)
            if isinstance(res, dict) and is_empty(res):
                continue
            if isinstance(res, list) and is_empty_list(res):
                continue
            if isinstance(res, list):
                results = []
                for res_el in res:
                    slots = res_el[RES_SLOTS]
                    scope = [s[RES_ENTITY] for s in slots if is_builtin_entity(s[RES_ENTITY])]
                    resolved_slots = resolve_slots(text, slots, entities, language, scope)
                    results.append(parsing_result(text, intent=res_el[RES_INTENT], slots=resolved_slots))
                return results

            slots = res[RES_SLOTS]
            scope = [s[RES_ENTITY] for s in slots if is_builtin_entity(s[RES_ENTITY])]
            resolved_slots = resolve_slots(text, slots, entities, language, scope)
            return [parsing_result(text, intent=res[RES_INTENT], slots=resolved_slots)]
        return [empty_result(text)]
