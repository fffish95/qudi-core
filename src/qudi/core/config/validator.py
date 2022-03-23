# -*- coding: utf-8 -*-

"""
JSON (draft v7) validator for qudi YAML configurations that also fills in default values.
The corresponding JSON schema is defined in ".__schema.py".

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ['ValidationError', 'validate_config']

from typing import Mapping, Any
from jsonschema import validators as __validators
from jsonschema import Draft7Validator as __BaseValidator
from jsonschema import ValidationError
from .__schema import qudi_cfg_schema as __cfg_schema


def __set_defaults(validator, properties, instance, schema):
    for property, subschema in properties.items():
        if 'default' in subschema:
            try:
                instance.setdefault(property, subschema['default'])
            except AttributeError:
                pass

    for error in __BaseValidator.VALIDATORS['properties'](validator, properties, instance, schema):
        yield error


def __is_iterable(checker, instance):
    return (__BaseValidator.TYPE_CHECKER.is_type(instance, "array") or
            isinstance(instance, (set, frozenset, tuple)))


DefaultInsertionValidator = __validators.extend(
    validator=__BaseValidator,
    validators={'properties': __set_defaults},
    type_checker=__BaseValidator.TYPE_CHECKER.redefine("array", __is_iterable)
)


def validate_config(config: Mapping[str, Any]) -> None:
    DefaultInsertionValidator(__cfg_schema).validate(config)