# import pint
#
#
# def _parse_units(units: dict, ureg: pint.UnitRegistry | None = None, verbose: int = 0):
#     """
#        Convert a dict with string units to pint quantities.
#
#        Inputs:
#            - units: dict with {"variable_name": "unit"}
#            - ureg: optional: a pint UnitRegistry
#            - verbose: verbosity level (int; default: 0)
#
#        Returns
#            - parsed_units: dict with {"variable_name": pint Quantity}
#        """
#     parsed_units = {}
#     if units:
#         if ureg is None:
#             ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
#         for c in units:
#             try:
#                 parsed_units[c] = ureg.parse_expression(units[c])
#             except pint.UndefinedUnitError:
#                 if verbose > 0:
#                     print(f"[AutoFeat] WARNING: unit {units[c]} of column {c} was not recognized and will be ignored!")
#                 parsed_units[c] = ureg.parse_expression("")
#             parsed_units[c].__dict__["magnitude"] = 1.0
#     return parsed_units
#
# units_dict = {
#     "length": "meter",
#     "time": "second",
#     "mass": "kilogram"
# }
# ureg = pint.UnitRegistry(auto_reduce_dimensions=True, autoconvert_offset_to_baseunit=True)
# parsed_units = _parse_units(units_dict,ureg, verbose=1)
# print(parsed_units)
# pi_theorem_results = ureg.pi_theorem(parsed_units)
# print(pi_theorem_results)
# for i, r in enumerate(pi_theorem_results, 1):
#     print(f"[AutoFeat] Pi Theorem {i}: ", pint.formatter(r.items()))
#     # compute the final result by multiplying and taking the power of
#     cols = sorted(r)
#     # only use data points where non of the affected columns are NaNs
#     # not_na_idx = df[cols].notna().all(axis=1)
import numpy as np
import pandas as pd

# import re
#
# import sympy
#
#
# def colnames2symbols(c:str|int, i: int = 0) -> str:
#     # take a messy column name and transform it to something sympy can handle
#     # worst case: i is the number of the features
#     # has to be a string
#     c = str(c)
#     # should not contain non-alphanumeric characters
#     c = re.sub(r"\W+", "", c)
#     if not c:
#         c = f"x{i:03}"
#     elif c[0].isdigit():
#         c = "x" + c
#     return c
#
# x = colnames2symbols(c=3,i=0)
# y = sympy.symbols(x)
# print(y)

# import numpy as np
# def trans_operations(op):
#     if op == "sqrt":
#         op = np.sqrt
#     elif op == "sin":
#         op = np.sin

X = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
cols = list(np.array(list(X.columns)))
print(type(cols))