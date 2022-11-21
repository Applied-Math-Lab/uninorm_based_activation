# The contents of this file are licensed at your option under any and/or all of the following:
# WTFPL, CC0, Apache 2.0, MIT, BSD 3-clause, MPL 2.0, GPL2.0, GPL3.0, LGPL, CDDL1.0, EPL1.0.
# So pick your favorite license, do what you want, and have fun!

from typing import cast, Union, Dict, Mapping, Optional, Tuple, Any, List, Sequence, TextIO
import numpy as np
import pandas as pd
import scipy.io.arff as arff
import copy
import json
import datetime
import os
import sys
import csv


def sort_order(l: List[Any]) -> List[int]:
    return sorted(range(len(l)), key = l.__getitem__)


# Represents the meta data along a single axis of a Tensor
class MetaData(object):
    # axis specifies which axis this metadata is bound to.
    # (None indicates that one attribute applies to the whole tensor.)
    # names is a list of strings that describe each attribute.
    # cats is a list of (list of categories) for each attribute.
    # types is a list of strings that describe type hints (str, int, float, datetime64, etc)
    # (An empty list indicates a continuous attribute.)
    def __init__(self,
        axis: Optional[int],
        names: Optional[Sequence[str]] = None,
        cats: Optional[Sequence[Sequence[str]]] = None,
        types: Optional[Sequence[Union[str, None]]] = None,
    ) -> None:
        assert axis is None or axis >= 0, 'The axis must be canonicalized before it is passed to the MetaData constructor' # data.shape is needed to canonicalize, so it cannot be done here
        self.axis = axis
        self.names = list(names) if names else []
        if len(set(self.names)) < len(self.names):
            raise ValueError(f'Duplicate attribute names: {self.names}!')
        self.cats = [ list(c) for c in cats ] if cats else [ [] for _ in range(len(self.names)) ]
        self.types = list(types) if types else [ None for _ in range(len(self.names)) ]
        if axis is None and (len(self.names) > 1 or len(self.cats) > 1 or len(self.types) > 1):
            raise ValueError('Multiple attribute specifications were provided for the universal axis.')
        self.cat_to_enum: List[Dict[str, int]] = []
        for attr_cats in self.cats:
            t: Dict[str, int] = {}
            for i in range(len(attr_cats)):
                t[attr_cats[i]] = i
            self.cat_to_enum.append(t)

    # Marshalls the meta-data into plain-old-data
    def marshall(self) -> Mapping[str, Any]:
        return {
            'axis': self.axis,
            'names': self.names,
            'cats': self.cats,
            'types': self.types,
        }

    # Unmarshalls the meta data from plain-old-data
    @staticmethod
    def unmarshall(ob: Mapping[str, Any]) -> 'MetaData':
        return MetaData(ob['axis'], ob['names'], ob['cats'], ob['types'])

    # Returns a deep copy of this object
    def deepcopy(self) -> 'MetaData':
        return MetaData(
            self.axis,
            copy.deepcopy(self.names),
            copy.deepcopy(self.cats),
            copy.deepcopy(self.types)
        )

    # Returns True iff self has the same values as other.
    # If strict is False, then the ordering of columns and categorical values will be ignored.
    def is_equal(self, other: 'MetaData', strict: bool=True) -> bool:
        if self is other: return True
        self.complete(max(len(self.names), len(self.cats), len(self.types)))
        other.complete(max(len(other.names), len(other.cats), len(other.types)))
        if len(self.names) != len(other.names): return False
        if strict:
            if self.names != other.names: return False
            if self.cats != other.cats: return False
        else:
            if set(self.names) != set(other.names): return False
            for i in range(len(self.cats)):
                if set(self.cats[i]) != set(other.cats[i]): return False
        return True

    # Ensures that the meta data is fully specified
    def complete(self, size: int) -> None:
        while len(self.names) < size:
            self.names.append('attr_' + str(len(self.names)))
        while len(self.cats) < size:
            self.cats.append([])
        while len(self.types) < size:
            self.types.append(None)

    # Returns metadata sorted with the specified order
    def sort(self, order: List[int]) -> 'MetaData':
        self.complete(len(order))
        newcats = [self.cats[i] for i in order]
        newnames = [self.names[i] for i in order]
        newtypes = [self.types[i] for i in order]
        return MetaData(self.axis, newnames, newcats, newtypes)

    # Returns the numerical representation of the specified string given the specified attribute
    def to_val(self, attr: int, s: str) -> int:
        if self.axis is None:
            attr = 0 # if axis is None, the metadata applies to every axis
        return self.cat_to_enum[attr][s]

    # Returns a string representation of the specified value
    def to_str(self, attr: int, val: float) -> str:
        if self.axis is None:
            attr = 0 # if axis is None, the metadata applies to every axis
        if attr >= len(self.cats):
            return str(val) # Continuous
        d = self.cats[attr]
        if len(d) == 0:
            return str(val) # Continuous
        elif np.isnan(val):
            return 'NaN'
        else:
            cat_index = int(val)
            if cat_index >= 0 and cat_index < len(d):
                return d[int(val)] # Categorical
            else:
                raise ValueError(f'{val} is not a valid index into the list of {len(d)} categories for attribute {attr}')

    # A helper for methods that reduce a Tensor
    def reduce(self, operation_name: str, axis:Optional[int]) -> 'MetaData':
        if axis is None:
            return MetaData(None, [operation_name])
        else:
            assert axis >= 0, 'The axis must be canonicalized before calling this method'
            if axis == self.axis:
                return MetaData(None, [operation_name])
            elif self.axis is None or axis > self.axis:
                return self.deepcopy()
            else:
                newmeta = self.deepcopy()
                newmeta.axis -= 1 # type: ignore
                return newmeta

    # Implements the [] operator to support slicing
    def __getitem__(self, args: Any) -> 'MetaData':
        if isinstance(args, tuple): # Tuple
            newaxis = self.axis
            newcats = self.cats
            newnames = self.names
            newtypes = self.types
            i = 0
            for a in args:
                if self.axis is not None and i < self.axis:
                    if not isinstance(a, slice) and not isinstance(a, list):
                        assert(newaxis is not None)
                        newaxis -= 1
                elif self.axis is not None and i == self.axis:
                    if isinstance(a, slice): # A slice of attributes
                        newcats = newcats[a]
                        newnames = newnames[a]
                        newtypes = newtypes[a]
                    elif isinstance(a, list):
                        newcats = [self.cats[j] for j in a]
                        newnames = [self.names[j] for j in a]
                        newtypes = [self.types[j] for j in a]
                    else:
                        newcats = [self.cats[a]]
                        if len(newnames) > a:
                            newnames = [newnames[a]]
                        else:
                            newnames = []
                        if len(newtypes) > a:
                            newtypes = [newtypes[a]]
                        else:
                            newtypes = []
                        newaxis = None
                else:
                    break
                i += 1
            return MetaData(newaxis, newnames, newcats, newtypes)
        elif isinstance(args, slice): # Slice axis 0
            if self.axis == 0:
                return MetaData(self.axis, self.names[args], self.cats[args], self.types[args])
            else:
                return self # Nothing to change
        elif isinstance(args, list): # List for axis 0
            if self.axis is not None and self.axis > 0:
                return self
            elif self.axis is not None and self.axis == 0:
                #return MetaData(0, self.names[args], [self.cats[args]], [self.types[args]])
                raise ValueError('Sorry, indexing the meta axis with a list is not yet supported')
            else:
                return self # Nothing to change
        else: # Index axis 0
            if self.axis is not None and self.axis > 0:
                return MetaData(self.axis - 1, self.names, self.cats, self.types) # Drop first axis
            elif self.axis is not None and self.axis == 0:
                return MetaData(None, self.names[args], [self.cats[args]], [self.types[args]]) # List of one element
            else:
                return self # Nothing to change

    # Returns a string representation of this object
    def __str__(self) -> str:
        s = 'MetaData(axis=' + str(self.axis) + '):\n'
        for i in range(len(self.cats)):
            if len(self.names) > i: s += self.names[i]
            else: s += 'untitled'
            s += ': '
            d = self.cats[i]
            if len(d) == 0: s += 'Continuous'
            elif len(d) < 16: s += str(d)
            else:
                s += '[' + d[0]
                for i in range(1, 16):
                    s += ',' + d[i]
                s += ', ... (' + str(len(d)) + ' total)]'
            s += '\n'
        return s

    # Returns true iff the specified attribute is continuous (as opposed to categorical)
    def is_continuous(self, attr: int) -> bool:
        if self.axis is None:
            attr = 0
        if attr >= len(self.cats):
            return True
        else:
            return len(self.cats[attr]) == 0

    # Returns the number of categories in a categorical attribute
    def categories(self, attr: int) -> int:
        if self.axis is None:
            attr = 0
        return len(self.cats[attr])

    # Returns whether this meta contains type hints
    def has_type_hints(self) -> bool:
        for t in self.types:
            if t is not None:
                return True
        return False

    # Returns a list of tuples that specify the original attribute and value corresponding
    # with the attributes in a one-hot encoding of a tensor with this meta data
    def one_hot_map(self) -> List[Tuple[int, int]]:
        cat_map: List[Tuple[int, int]] = []
        for i in range(len(self.names)):
            cat_map += [(i, j) for j in range(1 if len(self.cats[i]) < 3 else len(self.cats[i]))]
        return cat_map












# A numpy array with some meta data attached to one axis
class Tensor():

    # data is a float or numpy array
    # meta is a MetaData object that describes the data
    def __init__(self,
        data: Any,
        meta: MetaData
    ) -> None:
        self.data: Any = data
        self.meta = meta

        # Check that the data and metadata line up
        if self.meta.axis is not None:
            if not isinstance(self.data, np.ndarray):
                raise ValueError(f'Expected a np.ndarray. Got a {type(data)}.')
            if self.meta.axis >= len(self.data.shape):
                raise ValueError(f'Data has only {len(self.data.shape)} axes, but caller specified to attach meta data to axis {self.meta.axis}')

        # Ensure the meta data aligns with the size of the specified axis
        attr_count = 1 if self.meta.axis is None else self.data.shape[self.meta.axis]
        if len(self.meta.names) > attr_count or len(self.meta.cats) > attr_count:
            raise ValueError(f'{len(self.meta.names)} names and {len(self.meta.cats)} category lists were provided for axis {self.meta.axis}, which only has a size of {attr_count}')
        while len(self.meta.names) < attr_count:
            self.meta.names.append(f'attr_{len(self.meta.names)}')
        while len(self.meta.cats) < attr_count:
            self.meta.cats.append([])
            self.meta.cat_to_enum.append({})

    # Marshalls this tensor into plain-old-data
    def marshall(self) -> Mapping[str, Any]:
        return {
            'meta': self.meta.marshall(),
            'data': self.data.tolist(),
        }

    # Unmarshalls this tensor from plain-old-data
    @staticmethod
    def unmarshall(ob: Mapping[str, Any]) -> 'Tensor':
        return Tensor(np.array(ob['data']), MetaData.unmarshall(ob['meta']))

    # Returns a deep copy of this tensor
    def deepcopy(self) -> 'Tensor':
        return Tensor(np.copy(self.data), self.meta.deepcopy())

    # Returns the positive equivalent index of the specified axis
    def axis_index(self, axis: int) -> int:
        return axis if axis >= 0 else len(self.data.shape) + axis

    # This overloads the [] operator
    def __getitem__(self, args: Any) -> 'Tensor':
        if isinstance(args, tuple):
            attr = self.meta.axis or 0
            if isinstance(args[attr], str): # If the attribute is specified by name
                attr_index = self.meta.names.index(args[attr])
                new_slices = tuple([attr_index if i == attr else args[i] for i in range(len(args))])
                newdata = self.data[new_slices]
                newmeta = self.meta[new_slices]
                return Tensor(newdata, newmeta)
            elif isinstance(args[attr], list) and len(args[attr]) > 0 and isinstance(args[attr][0], str): # If attributes are specified as a list of names
                attr_names = args[attr]
                attr_indexes = [self.meta.names.index(name) for name in attr_names]
                new_slices = tuple([attr_indexes if i == attr else args[i] for i in range(len(args))])
                newdata = self.data[new_slices]
                newmeta = self.meta[new_slices]
                return Tensor(newdata, newmeta)
            else:
                return Tensor(self.data[args], self.meta[args])
        else:
            return Tensor(self.data[args], self.meta[args])

    # Returns the number of axes in this tensor
    def rank(self) -> int:
        if isinstance(self.data, np.ndarray):
            return len(self.data.shape)
        else:
            return 0

    # Generates a string represention of the matrix made of the last two axes in a tensor.
    # coords should be a full-rank list of integers for the whole tensor.
    # The last 2 elements in coords will be used as an in-place buffer.
    def matrix_str(self, coords: List[int]) -> str:
        s = ''
        dims = self.rank()
        attr = self.meta.axis or 0

        # Measure column widths
        colwidths = [0] * self.data.shape[dims - 1]
        for coords[dims - 2] in range(self.data.shape[dims - 2]):
            for coords[dims - 1] in range(self.data.shape[dims - 1]):
                v = self.meta.to_str(coords[attr], self.data[tuple(coords)])
                colwidths[coords[dims - 1]] = max(colwidths[coords[dims - 1]], len(v))

        # Column names
        if attr == dims - 1 and self.meta.names:
            s += '  '
            for i in range(self.data.shape[dims - 1]):
                colname = self.meta.names[i]
                if len(colname) > colwidths[i] + 1: colname = colname[:colwidths[i] + 1]
                if len(colname) < colwidths[i] + 1: colname = ' ' * (colwidths[i] + 1 - len(colname)) + colname
                s += colname + ' '
            s += '\n'

        # Make the matrix
        for coords[dims - 2] in range(self.data.shape[dims - 2]):
            if coords[dims - 2] == 0: s += '['
            else: s += ' '

            # Row names
            if attr == dims - 2:
                max_row_name_len = 12
                rowname = self.meta.names[coords[dims - 2]]
                if len(rowname) > max_row_name_len: rowname = rowname[:max_row_name_len]
                if len(rowname) < max_row_name_len: rowname = ' ' * (max_row_name_len - len(rowname)) + rowname
                s += rowname + ':'

            # Row data
            s += '['
            for coords[dims - 1] in range(self.data.shape[dims - 1]):
                v = self.meta.to_str(coords[attr], self.data[tuple(coords)])
                if coords[dims - 1] > 0:
                    s += ', '
                s += ' ' * (colwidths[coords[dims - 1]] - len(v)) + v # Pad on left side with spaces
            if coords[dims - 2] == self.data.shape[dims - 2] - 1: s += ']]\n';
            else: s += ']\n';
        return s

    # Make a human-readable string representation of this tensor
    def __str__(self) -> str:
        if isinstance(self.data, np.ndarray):
            if self.rank() == 0: # Single element
                # Actually, I don't think this case ever occurs because numpy doesn't allow zero-rank tensors
                # Oh well, doesn't hurt to support it just in case they ever decide to fix that.
                assert self.meta.axis is None, 'non-existent axis'
                s = ''
                s += self.meta.names[0] + ':'
                return s + self.meta.to_str(0, self.data)
            elif self.rank() == 1: # Vector
                if self.meta.axis is None:
                    s = self.meta.names[0] + ':'
                else:
                    s = ''
                s += '['
                if self.data.shape[0] > 0:
                    if self.meta.axis == 0:
                        s += self.meta.names[0] + ':'
                    s += self.meta.to_str(0, self.data[0])
                for i in range(1, self.data.shape[0]):
                    s += ', '
                    if self.meta.axis == 0:
                        s += self.meta.names[i] + ':'
                    s += self.meta.to_str(i, self.data[i])
                return s + ']'
            elif self.rank() == 2: # Matrix
                return self.matrix_str([0, 0])
            else:
                coords = [0] * self.rank()
                keepgoing = True

                # Visit all coordinates (not including the last 2)
                s = ''
                attr = 0
                while keepgoing:

                    # Opening brackets
                    bracks = 0
                    for i in range(len(coords) - 2):
                        if coords[len(coords) - 3 - i] == 0:
                            s += '['
                            bracks += 1
                        else: break

                    # # Axis label
                    # if self.meta.axis == len(self.data.shape) - 3 - bracks:
                    #     s += self.meta.cats[attr] + ':'
                    #     attr += 1

                    # Print the rank-2 slice
                    s += '\n'
                    s += self.matrix_str(coords)

                    # Closing brackets
                    for i in range(len(coords) - 2):
                        dim = len(coords) - 3 - i
                        if coords[dim] == self.data.shape[dim] - 1: s += ']'
                        else: break
                    s += '\n'

                    # Increment the coordinates
                    for revdim in range(len(coords) - 2):
                        dim = len(coords) - 3 - revdim
                        coords[dim] += 1
                        if coords[dim] >= self.data.shape[dim]:
                            coords[dim] = 0
                            if dim == 0:
                                keepgoing = False
                        else:
                            break
                return s
        else: # Single value
            assert self.meta.axis is None, 'meta data on missing axis'
            s = ''
            s += self.meta.names[0] + ':'
            return s + self.meta.to_str(0, self.data)

    # Use the same string representation even when it is part of a collection
    __repr__ = __str__

    # Gets the value at the specified coordinates.
    # If the value is categorical, returns its enumeration as a float.
    def get_float(self, *coords: int) -> float:
        return cast(float, self.data[coords].item())

    # Gets the value at the specified coordinates.
    # If the value is continuous, returns a string representation of its value.
    def get_string(self, *coords: int) -> str:
        attr = self.meta.axis or 0
        return self.meta.to_str(coords[attr], self.data[coords])

    # Returns the element at the specified coordinates.
    def get(self, *coords: Union[int, str]) -> Union[float, str]:
        attr = self.meta.axis or 0
        for i, x in enumerate(coords):
            if isinstance(x, str) and i != attr:
                raise ValueError('only the meta access can be accessed by string')
        int_coords = [
            self.meta.names.index(x) if isinstance(x, str) else x
            for x in coords
        ]
        if self.meta.is_continuous(int_coords[attr]):
            v = self.get_float(*int_coords)
            return v
        else:
            s = self.get_string(*int_coords)
            return s

    # Sets a categorical value by string.
    # If the string does not match one of the categories specified in the meta data,
    # or if the attribute is not categorical, then this will raise an error.
    def set_string(self, coords: Tuple[int, ...], val: str) -> None:
        attr = self.meta.axis or 0
        d = self.meta.to_val(coords[attr], val)
        self.data[coords] = d

    # Sets the specified element to the specified value.
    def set(self, coords: Tuple[int, ...], val: Union[float, str]) -> None:
        if isinstance(val, str):
            self.set_string(coords, val)
        else:
            self.data[coords] = val

    # Sets a value by string, adding a new category if necessary to accomodate it.
    # Raises an error if the specified attribute is not categorical.
    def insert_string(self, coords: Tuple[int, ...], s: str) -> None:
        s = str(s)
        i = coords[self.meta.axis] if self.meta.axis is not None else 0
        to_enum = self.meta.cat_to_enum[i]
        if s in to_enum: # If this is already an existing category...
            self.data[coords] = to_enum[s]
        else:
            # Add a new category
            attr_cat = self.meta.cats[i]
            n = len(attr_cat)
            attr_cat.append(s) # Add to the forward mapping
            to_enum[s] = n # Add to the reverse mapping
            self.data[coords] = n

    # Sets the specified element to the specified value.
    def insert(self, coords: Tuple[int, ...], val: Union[float, str]) -> None:
        if isinstance(val, str):
            self.insert_string(coords, val)
        elif isinstance(val, bytes):
            self.insert_string(coords, str(val))
        elif isinstance(val, pd.Timestamp):
            self.insert_string(coords, pd.to_datetime(val).strftime('%Y-%m-%d'))
        else:
            self.data[coords] = val

    # Adopts the supplied MetaData object, remapping values as needed to conform with the new MetaData.
    # If a categorical value is encountered that is not found in the new MetaData object,
    # behavior depends on allow_unrecognized_cat_vals. It sets the value to nan if true, else throws.
    def remap_cat_vals(self, template_meta: MetaData, allow_unrecognized_cat_vals: bool) -> None:
        # Check for compatibility
        if self.meta.axis != template_meta.axis:
            raise ValueError('Attributes on different axes')
        if self.data.shape[self.meta.axis or 0] != len(template_meta.cats):
            raise ValueError('Different sizes on the meta axes')
        for attr in range(self.data.shape[self.meta.axis or 0]):
            if self.meta.is_continuous(attr) != template_meta.is_continuous(attr):
                raise ValueError('Mismatching data types (one is continuous and one is categorical) in attribute ' + str(attr))

        # Remap the categorical values
        n = self.meta.axis or 0
        for attr in range(self.data.shape[n]):
            if self.meta.is_continuous(attr):
                continue
            m: Dict[int, int] = {}
            for in_val in range(len(self.meta.cats[attr])):
                s = self.meta.cats[attr][in_val]
                if allow_unrecognized_cat_vals:
                    out_val = template_meta.cat_to_enum[attr][s] if s in template_meta.cat_to_enum[attr] else np.nan
                else:
                    out_val = template_meta.cat_to_enum[attr][s]
                m[in_val] = out_val
            attr_slice_tuple = (slice(None),) * n + cast(Tuple[Any], (attr,)) + (slice(None),) * (self.rank() - n - 1)
            slice_of_self_to_remap = self.data[attr_slice_tuple]
            with np.nditer(slice_of_self_to_remap, op_flags = ['readwrite']) as it:
                for x in it:
                    x[...] = np.nan if np.isnan(x) else m[int(x)]
        self.meta = template_meta

    # Converts this tensor to raw Python structures
    def to_list(self) -> Any:
        l: List[Any] = []
        if self.rank() > 1:
            for i in range(self.data.shape[0]):
                l.append(self[i].to_list())
        elif self.rank() == 1:
            for i in range(self.data.shape[0]):
                l.append(self.get(i))
        else:
            return self.get(0)
        return l

    # Extends this tensor along the specified axis.
    # Extending on the axis with meta data is not currently supported.
    # If fill_value is None, the new rows will be uninitialized.
    def extend_inplace(self, axis: int, amount: int, fill_value: Optional[float] = None) -> None:
        if amount <= 0:
            if amount == 0: return
            else: raise ValueError('Negative extensions are not allowed. Just use slicing')
        axis = self.axis_index(axis)
        if axis == self.meta.axis: raise ValueError('Sorry, extending along the meta axis is not allowed')
        extension_shape = self.data.shape[ : axis] + (amount,) + self.data.shape[axis + 1 : ]
        extension = np.empty(extension_shape)
        if fill_value is not None:
            extension.fill(fill_value)
        self.data = np.concatenate([self.data, extension], axis)

    # Normalizes (in place) all of the non-categorical attributes to fall in the range [0., 1.]
    def normalize_inplace(self) -> None:
        for i in range(self.data.shape[self.meta.axis or 0]):
            if self.meta.is_continuous(i):
                self.meta.names[i] = 'n_' + self.meta.names[i]
                slice_list_in: List[Any] = [slice(None)] * len(self.data.shape)
                slice_list_in[self.meta.axis or 0] = i
                lo = self.data[tuple(slice_list_in)].min()
                hi = self.data[tuple(slice_list_in)].max()
                self.data[tuple(slice_list_in)] -= lo
                self.data[tuple(slice_list_in)] *= (1.0 / (hi - lo))

    # Normalizes all of the non-categorical attributes to fall in the range [0., 1.]
    def normalize(self) -> 'Tensor':
        c = self.deepcopy()
        c.normalize_inplace()
        return c

    # Encodes all of the categorical attributes with a one-hot encoding
    def one_hot(self) -> 'Tensor':
        if self.meta.axis is None:
            raise NotImplementedError('Sorry, I have not thought about this case yet')

        # Build list of new attribute names, and count the new size
        newnames = []
        newsize = 0
        for i in range(self.data.shape[self.meta.axis]):
            if self.meta.is_continuous(i) or self.meta.categories(i) == 2:
                newsize += 1
                newnames.append(self.meta.names[i])
            else:
                newsize += self.meta.categories(i)
                for j in range(self.meta.categories(i)):
                    newnames.append(self.meta.cats[i][j] + '_' + self.meta.names[i])

        # Allocate buffer for new data
        newshape = self.data.shape[:self.meta.axis] + (newsize,) + self.data.shape[self.meta.axis + 1:]
        newdata = np.zeros(newshape)

        # Transform the data
        o = 0
        for i in range(self.data.shape[self.meta.axis]):
            slice_list_in: List[Any] = [slice(None)] * len(self.data.shape)
            slice_list_in[self.meta.axis] = i
            if self.meta.is_continuous(i):
                # Copy straight over
                slice_list_out: List[Any] = [slice(None)] * len(self.data.shape)
                slice_list_out[self.meta.axis] = o
                newdata[tuple(slice_list_out)] = self.data[tuple(slice_list_in)]
                o += 1
            elif self.meta.categories(i) == 2:
                # Copy straight over, but replace NaNs with 0.5
                slice_list_out = [slice(None)] * len(self.data.shape)
                slice_list_out[self.meta.axis] = o
                attr_vals = self.data[tuple(slice_list_in)]
                attr_vals[np.where(np.isnan(attr_vals))] = 0.5
                newdata[tuple(slice_list_out)] = attr_vals
                o += 1
            else:
                if self.meta.categories(i) > 300:
                    raise ValueError(f'Attempted to one-hot encode {self.meta.names[i]}, an attribute with more than 300 categorical values!')

                # Convert to a one-hot encoding
                olddata = self.data[tuple(slice_list_in)] # isolate just the relevant attribute
                it = np.nditer(olddata, flags=['multi_index']) # iterate over all the elements
                while not it.finished:
                    categorical_value = -1 if np.isnan(it[0]) else int(it[0]) # get the categorical value
                    if categorical_value >= 0 and categorical_value < self.meta.categories(i): # if the value is valid
                        hotindex = it.multi_index[:self.meta.axis] + (o + categorical_value,) + it.multi_index[self.meta.axis + 1:]
                        newdata[hotindex] = 1.
                    it.iternext()
                o += self.meta.categories(i)

        # Make a tensor with the transformed data
        newmeta = MetaData(self.meta.axis, newnames)
        return Tensor(newdata, newmeta)

    # Reorders slices along the specified axis according to the specified order
    def reorder(self, axis: int, sort_order: List[int]) -> 'Tensor':
        axis = self.axis_index(axis)
        assert(len(sort_order) == self.data.shape[axis]), 'order data wrong length'

        # Sort the data
        slice_list_out: List[Any] = []
        for i in range(len(self.data.shape)):
            if i == axis:
                slice_list_out.append(sort_order)
            else:
                slice_list_out.append(slice(None))
        newdata = self.data[tuple(slice_list_out)]

        # Sort the meta data
        if axis == self.meta.axis:
            newmeta = self.meta.sort(sort_order)
        else:
            newmeta = self.meta
        return Tensor(newdata, newmeta)

    # coords should have the same size as the shape of this tensor.
    # coords should contain exactly one element equal to -1.
    # The -1 indicates the axis to sort along.
    # Returns a tensor sorted along the axis specified with a -1 in coords.
    def sort(self, coords: Tuple[int, ...]) -> 'Tensor':
        if len(coords) != len(self.data.shape):
            raise ValueError('Expected coords to contain ' + str(len(self.data.shape)) + ' elements, with one -1 to indicate the axis to sort on.')

        # Identify the values that need to be sorted from the specified coordinates
        slice_list_in: List[Any] = []
        axis = -1
        for i in range(len(coords)):
            if coords[i] == -1:
                slice_list_in.append(slice(None))
                if axis > -1: raise ValueError('Expected only one value in coords to contain -1')
                axis = i
            else:
                slice_list_in.append(coords[i])

        # Determine the sort order
        sort_me = self.data[tuple(slice_list_in)]
        attr = coords[self.meta.axis or 0]
        if attr != -1 and not self.meta.is_continuous(attr):
            cats = self.meta.cats[attr]
            no_sort_me_instead = [cats[int(x)] for x in sort_me]
            sort_order = np.array(sorted(range(len(no_sort_me_instead)), key = no_sort_me_instead.__getitem__))
        else:
            sort_order = np.argsort(sort_me, 0)
        return self.reorder(axis, sort_order.tolist())

    # Assumes self is a 2d tensor with metadata on axis 1.
    # Returns an tensor with the dates in sorted order with exactly one row for each day.
    # Missing categorical values will be carried over from the previous entry.
    # Missing continuous values will be interpolated.
    def fill_missing_dates(self, col_name: str='date', fmt: str='%Y-%m-%d') -> 'Tensor':
        if self.rank() != 2: raise ValueError('Expected a rank 2 tensor')
        if self.meta.axis != 1: raise ValueError('Expected metadata along axis 1')

        # Sort by date
        date_index = self.meta.names.index(col_name)
        sorted = self.sort((-1, date_index))

        # Count the number of days in the specified range
        start_date = datetime.datetime.strptime(sorted.get_string(0, date_index), fmt)
        last_date = datetime.datetime.strptime(sorted.get_string(-1, date_index), fmt)
        delta = datetime.timedelta(days = 1)
        rows: int = 0
        dt = start_date
        while dt <= last_date:
            rows += 1
            dt += delta
        if rows == sorted.data.shape[0]:
            return sorted

        # Interpolate missing dates
        j: int = 0 # j is the source row
        res = Tensor(np.zeros((rows, sorted.data.shape[1])), sorted.meta)
        dt = start_date # dt is the date of the destination row
        for i in range(rows): # i is the destination row
            res.data[i] = sorted.data[j] # copy from source to destination
            prev_date = datetime.datetime.strptime(res.get_string(i, date_index), fmt)
            if dt > prev_date: # Is interpolation needed here?
                next_row = min(sorted.data.shape[0] - 1, j + 1)
                next_date = datetime.datetime.strptime(sorted.get_string(next_row, date_index), fmt)
                for k in range(res.data.shape[1]):
                    if res.meta.is_continuous(k):
                        # Interpolate the value (Note: there seems to be some rounding issues in the next line)
                        res.data[i, k] = ((dt - prev_date) * sorted.data[next_row, k] + (next_date - dt) * sorted.data[j, k]) / (next_date - prev_date)
                res.insert_string((i, date_index), dt.strftime(fmt))

            # Advance
            dt += delta # Advance the destination date
            while j + 1 < sorted.data.shape[0] and datetime.datetime.strptime(sorted.get_string(j + 1, date_index), fmt) <= dt:
                j += 1 # Advance the source row past the destination date
        if res.data.shape[0] < self.data.shape[0]:
            raise ValueError('Made it smaller? Were there multiple values per day?')
        return res

    # Assumes the specified column contains dates in YY-MM-DD format.
    # Assumes the specified column contains at least two consecutive dates followed by NaNs.
    # Replaces all the NaNs in that column by extrapolating the last interval.
    def extrapolate_dates_inplace(self, col_name: str='date', fmt: str='%Y-%m-%d') -> None:
        if self.rank() != 2: raise ValueError('Expected a rank 2 tensor')
        if self.meta.axis != 1: raise ValueError('Expected metadata along axis 1')
        date_col = self.meta.names.index(col_name)
        fill_start: int = next((i for i in reversed(range(self.data.shape[0])) if not np.isnan(self.data[i, date_col])), None) # type: ignore
        if fill_start < 2:
            raise ValueError('Not enough known dates to determine the interval')
        a = datetime.datetime.strptime(self.get_string(fill_start - 2, date_col), fmt)
        b = datetime.datetime.strptime(self.get_string(fill_start - 1, date_col), fmt)
        delta = b - a
        for i in range(fill_start, self.data.shape[0]):
            prev = datetime.datetime.strptime(self.get_string(i - 1, date_col), fmt)
            self.insert_string((i, date_col), (prev + delta).strftime(fmt))

    # Convert all dates in the "date" column from US to ISO format.
    # Assumes each date is unique.
    def us_to_iso_dates_inplace(self) -> None:

        # Check assumptions
        if self.rank() != 2: raise ValueError('Expected a rank 2 tensor')
        if self.meta.axis != 1: raise ValueError('Expected metadata along axis 1')

        # Convert the dates
        date_index = self.meta.names.index('date')
        for i in range(self.data.shape[0]):
            val = int(self.data[i, date_index])
            old_str = self.get_string(i, date_index)
            d = datetime.datetime.strptime(old_str, '%m/%d/%Y')
            new_str = d.strftime('%Y/%m/%d')
            self.meta.cats[date_index][val] = new_str

    # Converts this tensor to a list of dicts
    def to_list_of_dict(self, drop_nans: bool = True) -> List[Dict[str, Any]]:

        # Check assumptions
        if self.rank() != 2: raise ValueError('Expected a rank 2 tensor')
        if self.meta.axis != 1: raise ValueError('Expected metadata along axis 1')

        # Make the list of dictionaries
        obs = []
        for i in range(self.data.shape[0]):
            d = {}
            for j in range(self.data.shape[1]):
                if not drop_nans or not np.isnan(self.data[i, j]):
                    d[self.meta.names[j]] = self.get(i, j)
            obs.append(d)
        return obs

    # Converts this tensor to a list of list of values, and some meta data
    def to_list_of_list(self) -> Tuple[List[List[Any]], List[Dict[str, Union[str, None]]]]:

        # Check assumptions
        if self.rank() != 2: raise ValueError('Expected a rank 2 tensor')
        if self.meta.axis != 1: raise ValueError('Expected metadata along axis 1')

        # Make the list of lists
        obs = []
        for i in range(self.data.shape[0]):
            l = []
            for j in range(self.data.shape[1]):
                l.append(self.get(i, j))
            obs.append(l)

        # Make the meta data
        meta = []
        for col in range(self.data.shape[1]):
            meta.append({ 'name': self.meta.names[col], 'type': self.meta.types[col] })

        return obs, meta

    # Converts this tensor to a JSON string
    def to_json(self) -> str:
        return json.dumps(self.marshall())

    # Writes this tensor to a file in JSON format
    def save_json(self, filename: str) -> None:
        b = bytes(self.to_json(), 'utf8')
        with open(filename, mode='wb+') as file:
            file.write(b)

    # Adds a dimension to self
    def expand_dims(self, axis: int) -> 'Tensor':
        axis = self.axis_index(axis)
        new_axis = self.meta.axis if self.meta.axis is None or axis > self.meta.axis else self.meta.axis + 1
        return Tensor(np.expand_dims(self.data, axis), MetaData(new_axis, self.meta.names, self.meta.cats))

    # Converts a Teddy Tensor to a Pandas DataFrame
    def to_pandas(self, index: Optional['Tensor'] = None) -> pd.DataFrame:
        if self.rank() != 2: raise ValueError('Sorry, this method only supports rank 2 tensors');
        if self.meta.axis != 1: raise ValueError('Expected meta data to be attached to columns');
        raw: Dict[str, Any] = {}
        for i in range(self.data.shape[1]):
            if self.meta.is_continuous(i):
                raw[self.meta.names[i]] = self.data[:, i]
            else:
                vals = [self.get_string(j, i) for j in range(self.data.shape[0])]
                raw[self.meta.names[i]] = pd.Categorical(vals, ordered = True)
        if index:
            assert index.rank() == 1, 'expected rank one data here'
            if index.meta.is_continuous(0):
                raw[index.meta.names[0]] = index.data
            else:
                vals = [index.get_string(j) for j in range(index.data.shape[0])]
                raw[index.meta.names[0]] = pd.Categorical(vals, ordered = True)
        df = pd.DataFrame(raw)
        if index:
            return df.set_index(index.meta.names[0], drop = True)
        else:
            return df

    # Transposes a 2-tensor
    def transpose(self) -> 'Tensor':
        assert len(self.data.shape) == 2, 'Transpose only works with matrices'
        new_data = np.transpose(self.data)
        new_meta = self.meta.deepcopy()
        new_meta.axis = new_meta.axis if new_meta.axis is None else 1 - new_meta.axis
        return Tensor(new_data, new_meta)

    # Wraps numpy.mean
    def mean(self, axis:Optional[int]=None) -> 'Tensor':
        return Tensor(np.nanmean(self.data, axis=axis), self.meta.reduce('mean', axis))

    # Wraps numpy.std
    def std(self, axis:Optional[int]=None) -> 'Tensor':
        return Tensor(np.nanstd(self.data, axis=axis), self.meta.reduce('std', axis))

    # Prints in CSV format with column names in the first row
    def print_csv(self, stream:TextIO=sys.stdout) -> None:
        if self.rank() != 2 or self.meta.axis != 1:
            raise ValueError('Expected a rank 2 tensor with meta axis 1')
        print(self.meta.names[0], end='', file=stream)
        for v in self.meta.names[1:]:
            print(',' + v, end='', file=stream)
        print(file=stream)
        for y in range(self.data.shape[0]):
            if not np.isnan(self.data[y, 0]):
                print(self.get_string(y, 0), end='', file=stream)
            for x in range(1, self.data.shape[1]):
                if np.isnan(self.data[y, x]):
                    print(',', end='', file=stream)
                else:
                    print(',' + self.get_string(y, x), end='', file=stream)
            print(file=stream)

    # Saves to a file in CSV format
    def save_csv(self, filename: str) -> None:
        with open(filename, 'w') as f:
            self.print_csv(stream=f)

    # Add a column of data
    def add_column(self, data: np.ndarray, name: str, cats: Optional[List[str]] = None) -> None:
        if self.rank() != 2 or self.meta.axis != 1:
            raise ValueError('Expected a rank 2 tensor with meta axis 1')
        if len(data.shape) != 1 or data.shape[0] != self.data.shape[0]:
            raise ValueError('Expected a compatible column vector')
        self.meta.names.append(name)
        self.meta.cats.append(cats or [])
        self.meta.types.append(None)
        self.data = np.concatenate([self.data, data[:, None]], axis=1)

    # Returns True iff all categorical values fall within the range specified in the metadata
    def check_cats(self, quiet:bool=False) -> bool:
        n = self.meta.axis or 0
        for j in range(len(self.meta.cats)):
            if not self.meta.is_continuous(j):
                slice_tuple = (slice(None),) * n + (slice(j, j + 1),) + (slice(None),) * (self.rank() - n - 1)
                s = self.data[slice_tuple]
                if np.nanmin(s) < 0:
                    if not quiet:
                        print(f'Out of range categorical value in attribute {self.meta.names[j]}')
                    return False
                if np.nanmax(s) >= len(self.meta.cats[j]):
                    if not quiet:
                        print(f'Out of range categorical value in attribute {self.meta.names[j]}')
                    return False
        return True




# Converts a Pandas DataFrame to Teddy Tensor
def from_pandas(df: Union[pd.DataFrame, pd.Series]) -> Tensor:
    if isinstance(df, pd.DataFrame):
        col_names = list(df.columns)
        names = [str(i) for i in col_names]
        cats: List[List[str]] = [[] for c in names]
        t = Tensor(np.zeros(df.shape), MetaData(1, names, cats))
        for c in range(len(col_names)):
            raw = list(df[col_names[c]])
            if len(raw) != t.data.shape[0]: raise ValueError('Expected ' + str(t.data.shape[0]) + ' values. Got ' + str(len(raw)))
            for r in range(len(raw)):
                t.insert((r, c), raw[r])
        return t
    elif isinstance(df, pd.Series):
        names = [df.name]
        cats = [[]]
        t = Tensor(np.zeros(len(df)), MetaData(0, names, cats))
        for i in range(len(df)):
            t.insert((i,), df[i])
        return t
    elif isinstance(df, np.ndarray): # Not really a Pandas type, but some Pandas methods return them
        if len(df.shape) == 1:
            meta = MetaData(None, ['vals'])
        else:
            meta = MetaData(1, ['attr_' + str(i) for i in range(df.shape[1])])
        return Tensor(df, meta)
    else:
        raise ValueError('Unsupported Pandas type: ' + str(type(df)))

# Initializes values from a list of tuples.
# Assumes the first row contains column names.
# Determines data types from whatever is in the second row.
def init_2d(data: List[Tuple[Any,...]]) -> Tensor:
    cats: List[List[str]] = []
    names: List[str] = []
    r0 = data[0]
    r1 = data[1]
    cat: List[bool] = [] # Used to check for consistency of types
    for col in range(len(r0)):
        colname = r0[col]
        if not isinstance(colname, str):
            raise ValueError('Expected the first row to contain column names as strings')
        names.append(colname)
        cats.append([])
        el = r1[col]
        if isinstance(el, str): cat.append(True)
        else: cat.append(False)
    t = Tensor(np.zeros((len(data) - 1, len(names))), MetaData(1, names, cats))
    for row in range(len(data) - 1):
        r = data[row + 1]
        if len(r) != len(names):
            raise ValueError('Row ' + str(row) + ' has ' + str(len(r)) + ' values. Expected ' + str(len(names)))
        for col in range(len(r)):
            if cat[col]:
                if not isinstance(r[col], str):
                    raise ValueError('Expected a string value at (' + str(row) + ',' + str(col) + ')')
                t.insert_string((row, col), r[col])
            else:
                if isinstance(r[col], str):
                    raise ValueError('Expected a numerical value at (' + str(row) + ',' + str(col) + ')')
                t.data[row, col] = r[col]
    return t

# Loads an ARFF file. Returns a Tensor.
def load_arff(filename: str) -> Tensor:
    data, meta = arff.loadarff(filename)
    cats: List[List[str]] = []
    for i in meta.types():
        cats.append([])
    t = Tensor(np.zeros((data.shape[0], len(data[0]))), MetaData(1, meta.names(), cats))
    for row in range(t.data.shape[0]):
        for col in range(t.data.shape[1]):
            if meta.types()[col] == 'nominal':
                t.insert_string((row, col), data[row][col].decode('us-ascii'))
            else:
                t.data[row, col] = data[row][col]
    return t

# Loads from a JSON format that is a list of tuples that redundantly repeat meta-data for every field
def from_list_of_dict(obs: List[Mapping[str, Any]]) -> Tensor:
    # extract column names (may have missing data in any single row)
    names = sorted(list(set(x for row in obs for x in row)))

    # extract the data into a tensor
    t = Tensor(np.full((len(obs), len(names)), np.nan), MetaData(1, names))
    for r, ob in enumerate(obs):
        for c, k in enumerate(names):
            numeric = False
            if k in ob:
                if ob[k] is None:
                    t.data[r, c] = np.nan
                elif isinstance(ob[k], str):
                    if numeric:
                        raise ValueError(f'The {k} attribute contains both strings and numbers')
                    t.insert_string((r, c), ob[k])
                elif np.isnan(ob[k]):
                    t.data[r, c] = np.nan
                else:
                    if len(t.meta.cats[c]) > 0:
                        raise ValueError(f'The {k} attribute contains strings and numbers')
                    t.data[r, c] = ob[k]
                    numeric = True
    assert t.check_cats(), 'cat problem'
    return t

# Loads from a JSON format that is a list of list of values
# An optional list of tuples may be supplied as metadata (type hints)
# Example without type hints:
# ([[ 2, 'dog' ],
#   [ 1, 'cat' ]], None)
# Example with type hints:
# ([[ 2, 'dog' ],
#   [ 1, 'cat' ]], [{'name': 'quantity', 'type':'float'}, {'name':'animal', 'type':'str'}])
def from_list_of_list(lol: Tuple[List[List[Any]], Optional[List[Mapping[str, Optional[str]]]]]) -> Tensor:
    obs = lol[0]
    cols = lol[1]

    # Extract metadata
    cats: List[List[str]] = []
    names: List[str] = []
    types: List[Union[str, None]] = []
    cat: List[int] = [] # Used to check for consistency of types
    if cols is not None:
        i = 0
        for attr in cols:
            name = attr['name'] or ''
            names.append(name)
            cats.append([])
            type = attr['type']
            types.append(type)
            if attr['type'] == 'str' or attr['type'] == 'datetime64' or isinstance(obs[0][i], str):
                cat.append(1)
            elif attr['type'] is not None and attr['type'].startswith('float'):
                cat.append(0)
            elif attr['type'] == None:
                cat.append(-1)
            else:
                raise ValueError(f'unexpected type {attr["type"]} for column {attr["name"]}')
            i += 1
    else:
        for i in range(len(obs[0])):
            cats.append([])
            names.append('attr_' + str(i))
            types.append(None)
            cat.append(-1)

    # Extract all the data
    t = Tensor(np.zeros((len(obs), len(names))), MetaData(1, names, cats, types))
    t.data[:] = np.nan
    row = 0
    for ob in obs:
        for col in range(min(len(ob), len(cat))):
            if ob[col] is not None:
                if cat[col] == 0:
                    t.data[row, col] = ob[col]
                elif cat[col] > 0:
                    t.insert_string((row, col), ob[col])
                else:
                    try:
                        t.data[row, col] = float(ob[col])
                        cat[col] = 0
                    except:
                        t.insert_string((row, col), ob[col])
                        cat[col] = 1

        row += 1
    return t

# Loads a CSV file. Returns a Tensor.
def load_csv(filename: str, column_names_in_first_row:bool=True) -> Tensor:
    # Load the file
    raw: List[List[str]] = []
    with open(filename, 'r') as f:
        iter = csv.reader(f)
        for line in iter:
            raw.append(list(line))

    # Parse the column names
    row = 0
    names: List[str] = []
    if column_names_in_first_row:
        if len(raw) < 1:
            raise ValueError('Expected a line of column names')
        names = raw[0]
        row += 1
    else:
        names = [ f'col_{i}' for i in range(len(raw[0])) ]

    # Determine the types. (It's numeric if every cell in the column can be interpreted as a number)
    types: List[bool] = []
    type_hints: List[Mapping[str, Optional[str]]] = []
    for c in range(len(names)):
        could_be_numeric = True
        for r in range(1 if column_names_in_first_row else 0, len(raw)):
            s = raw[r][c]
            if s == '' or s == 'NaN' or s == 'nan' or s == '?':
                pass
            else:
                try:
                    float(s)
                except:
                    could_be_numeric = False
                    break
        types.append(could_be_numeric)
        type_hints.append({ 'name': names[c], 'type': 'float' if could_be_numeric else 'str' })

    # Parse the data
    lol: List[List[Any]] = []
    while row < len(raw):
        vals: List[Any] = []
        for i, s in enumerate(raw[row]):
            if i >= len(names):
                break # Silently ignore extra columns
            if s == '' or s == 'NaN' or s == 'nan' or s == '?':
                vals.append(np.nan)
            elif types[i]:
                try:
                    vals.append(float(s))
                except:
                    raise ValueError(f'row {row+1}, col{i+1}: Expected float. Got {s}')
            else:
                vals.append(s)
        if len(vals) > 0:
            assert len(vals) == len(names), f'Expected more values in row {row+1}'
            lol.append(vals)
        row += 1
    t = from_list_of_list((lol, type_hints))
    if len(names) > 0:
        t.meta.names = names
    return t

def load_by_extension(filename: str) -> Tensor:
    _, ext = os.path.splitext(filename)
    if ext == '.arff':
        return load_arff(filename)
    elif ext == '.csv':
        return load_csv(filename)
    elif ext == '.json':
        return load_json(filename)
    else:
        raise ValueError('Unrecognized extension: ' + ext)

# Loads from a map of columns.
# For example, this input:
# { 'numbers': [1,2,3,4], 'letters': ['a', 'b', 'c', 'd'] }
# will produce a tensor with two columns and four rows.
# (Note that anything convertable to an np.array may be used in place of the lists)
def from_column_mapping(cols: Mapping[str, Any]) -> Tensor:
    arrs: List[np.array] = []
    names: List[str] = []
    for k in cols:
        names.append(k)
        if isinstance(cols[k][0], str):
            arr = np.zeros(len(cols[k]))
        else:
            arr = np.array(cols[k]).astype(np.float64)
        assert len(arr.shape) == 1, 'not a column'
        arrs.append(np.expand_dims(arr, 1))
    data = np.concatenate(arrs, 1)
    t = Tensor(data, MetaData(1, names))
    j = 0
    for k in cols:
        if isinstance(cols[k][0], str):
            kk = cols[k]
            for i in range(len(kk)):
                t.insert_string((i, j), kk[i])
        j += 1
    return t

# Loads from a JSON format that is a list of tuples that redundantly repeat meta-data for every field.
# Example:
#   [
#       ("date": "2018/02/18", "units": 23),
#       ("date": "2018/02/19", "units": 32)
#   ]
def load_json_list_of_dict(filename: str) -> Tensor:
    filecontents = None
    with open(filename, mode='rb') as file:
        filecontents = file.read()
    obs = json.loads(filecontents)
    return from_list_of_dict(obs)

# Parses JSON from a string
def parse_json(s: Union[str, bytes]) -> Tensor:
    return Tensor.unmarshall(json.loads(s))

# Loads from a JSON file
def load_json(filename: str) -> Tensor:
    filecontents = None
    with open(filename, mode='rb') as file:
        filecontents = file.read()
    return parse_json(filecontents)

# Modifies a list of tensors to all have uniform metadata.
# If template metadata is provided, the resulting tensors will have the template metadata.
# If no template is provided, the resulting tensors will have the metadata of the union of all the tensors.
def align(tensors: List['Tensor'], template: Optional[MetaData] = None) -> List['Tensor']:
    # Check for matching meta axes
    for i in range(1, len(tensors)):
        if tensors[i].meta.axis != tensors[0].meta.axis:
            raise ValueError('Tensors have meta data on different axes')
    if template is not None:
        if tensors[0].meta.axis != template.axis:
            raise ValueError('Tensors have meta data on different axes from the template')
    else:
        if(len(tensors) < 2):
            return tensors
    n = tensors[0].meta.axis or 0

    # Check that all the incoming categorical values fall within expected ranges
    for t in tensors:
        assert t.check_cats(), 'Attempted to align data with broken meta information'

    # Check if they are already aligned
    already_aligned = True
    for i in range(1, len(tensors)):
        if tensors[i].meta.names != tensors[0].meta.names or tensors[i].meta.cats != tensors[0].meta.cats:
            already_aligned = False
            break
    if template is not None:
        if tensors[0].meta.names != template.names or tensors[0].meta.cats != template.cats:
            already_aligned = False
    if already_aligned:
        return tensors

    # Get lists of all attribute names and categorical values
    if template is not None:
        all_names: List[str] = template.names
        all_cats: List[List[str]] = template.cats
    else:
        raw_names = [t.meta.names for t in tensors]
        all_names = list(set().union(*raw_names)) # type: ignore
        all_names.sort()
        all_cats = [
            list(set().union(*[(t.meta.cats[t.meta.names.index(attr_name)] if attr_name in t.meta.names else []) for t in tensors])) # type: ignore
            for attr_name in all_names
        ]
        for c in all_cats:
            c.sort()

    # Make an aligned slice for every attribue for every tensor
    slices: List[List[np.ndarray]] = [[] for t in tensors]
    for attr_dest_index, attr_name in enumerate(all_names):
        # Get a slice for this attribute in each tensor
        type_is_known = template is not None
        type_is_continuous = template.is_continuous(attr_dest_index) if template is not None else False
        for i, t in enumerate(tensors):
            if attr_name in t.meta.names:
                # Grab existing slice
                attr_src_index = t.meta.names.index(attr_name)
                if type_is_known and t.meta.is_continuous(attr_src_index) != type_is_continuous:
                    raise ValueError("Tensors have mismatching data types")
                type_is_known = True
                type_is_continuous = t.meta.is_continuous(attr_src_index)
                slice_tuple = (slice(None),) * n + (slice(attr_src_index, attr_src_index + 1),) + (slice(None),) * (t.rank() - n - 1)
                s = t.data[slice_tuple]
            else:
                # Make a new slice
                slice_shape = t.data.shape[:n] + (1,) + t.data.shape[n+1:]
                s = np.empty(slice_shape)
                s.fill(np.nan)
            slices[i].append(s)

        # Align categorical attributes
        if not type_is_continuous:
            # Map from old categorical values to new categorical values
            maps: List[Dict[int, int]] = [ {} for t in tensors ]
            for i, t in enumerate(tensors):
                if attr_name in t.meta.names:
                     attr_src_index = t.meta.names.index(attr_name)
                     for j, cat in enumerate(t.meta.cats[attr_src_index]):
                         maps[i][j] = all_cats[attr_dest_index].index(cat) if cat in all_cats[attr_dest_index] else -1

            # Remap the values of this categorical attribute
            for i in range(len(tensors)):
                with np.nditer(slices[i][attr_dest_index], op_flags = ['readwrite']) as iter:
                    for x in iter:
                        x[...] = np.nan if (np.isnan(x) or maps[i][int(x)] < 0) else maps[i][int(x)]

    # Concatenate the slices together to make the final list of tensors
    results: List['Tensor'] = []
    for i in range(len(tensors)):
        data = np.concatenate(slices[i], axis = tensors[0].meta.axis)
        meta = MetaData(tensors[0].meta.axis, all_names, all_cats)
        results.append(Tensor(data, meta))
    return results

# Concatenates multiple tensors along the specified axis
def concat(parts: List[Tensor], axis: int) -> Tensor:
    # Check assumptions
    if len(parts) == 0:
        raise ValueError('Expected at least one part')
    if len(parts) == 1:
        return parts[0]
    for i in range(1, len(parts)):
        if parts[i].meta.axis != parts[0].meta.axis:
            raise ValueError('Expected all parts to have the same meta axis')
        for j in range(len(parts[0].data.shape)):
            if j != axis and j != parts[0].meta.axis and parts[i].data.shape[j] != parts[0].data.shape[j]:
                raise ValueError('Expected all parts to have the same shape except along the joining axis')

    # Concatenate the parts
    if axis == parts[0].meta.axis:
        # Complete the metadata, so concatenating them will stay aligned
        for i in range(len(parts) - 1):
            parts[i].meta.complete(parts[i].data.shape[parts[i].meta.axis])

        # Fuse
        newcats: List[List[str]] = []
        newnames: List[str] = []
        for p in parts:
            newcats = newcats + p.meta.cats
            newnames = newnames + p.meta.names
        newmeta = MetaData(parts[0].meta.axis, newnames, newcats)
        newdata = np.concatenate([p.data for p in parts], axis = axis)
        return Tensor(newdata, newmeta)
    else:
        aligned_parts = align(parts)
        return Tensor(np.concatenate([p.data for p in aligned_parts], axis = axis), aligned_parts[0].meta)

# Joins two tensors to have the attributes of all of them.
# If any attributes have conflicting names, the tensor that occurs first in the list will override.
# If the tensors do not have the same sizes, they will be extended with nans.
def join(parts: List[Tensor]) -> Tensor:
    if len(parts) < 2:
        return parts[0]

    # Compute the new shape and drop redundant attributes
    new_shape = list(parts[0].data.shape)
    all_attr_names = set(parts[0].meta.names)
    for i in range(1, len(parts)):
        part = parts[i]
        if len(part.data.shape) != len(new_shape): raise ValueError('Different ranks')
        if part.meta.axis != parts[0].meta.axis: raise ValueError('Different meta axes')
        for j in range(len(new_shape)):
            if j == parts[0].meta.axis:
                continue
            new_shape[j] = max(new_shape[j], part.data.shape[j])
        unique_names = []
        for name in part.meta.names:
            if not name in all_attr_names:
                unique_names.append(name)
                all_attr_names.add(name)
        if len(unique_names) < len(part.meta.names):
            n = part.meta.axis or 0
            slices = cast(Tuple[Any], (slice(None),) * n) + cast(Tuple[Any], (unique_names,)) + cast(Tuple[Any], (slice(None),) * (parts[i].rank() - n - 1))
            parts[i] = part[slices]

    # Extend as needed
    for i in range(len(parts)):
        part = parts[i]
        for j in range(len(new_shape)):
            if j != part.meta.axis and part.data.shape[j] < new_shape[j]:
                part.extend_inplace(j, new_shape[j] - part.data.shape[j], np.nan)
        parts[i] = part

    # Concatenate
    return concat(parts, part.meta.axis or 0)
