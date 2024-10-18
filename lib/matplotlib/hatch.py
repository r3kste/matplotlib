"""Contains classes for generating hatch patterns."""

import numpy as np

from matplotlib import _api
from matplotlib.path import Path


class HatchPatternBase:
    """The base class for a hatch pattern."""
    pass


class HorizontalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('-') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = 0.0
        vertices[0::2, 1] = steps
        vertices[1::2, 0] = 1.0
        vertices[1::2, 1] = steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class VerticalHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int((hatch.count('|') + hatch.count('+')) * density)
        self.num_vertices = self.num_lines * 2

    def set_vertices_and_codes(self, vertices, codes):
        steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
                                      retstep=True)
        steps += stepsize / 2.
        vertices[0::2, 0] = steps
        vertices[0::2, 1] = 0.0
        vertices[1::2, 0] = steps
        vertices[1::2, 1] = 1.0
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class NorthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('/') + hatch.count('x') + hatch.count('X')) * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class SouthEastHatch(HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines = int(
            (hatch.count('\\') + hatch.count('x') + hatch.count('X'))
            * density)
        if self.num_lines:
            self.num_vertices = (self.num_lines + 1) * 2
        else:
            self.num_vertices = 0

    def set_vertices_and_codes(self, vertices, codes):
        steps = np.linspace(-0.5, 0.5, self.num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


class Shapes(HatchPatternBase):
    filled = False

    def __init__(self, hatch, density):
        if self.num_rows == 0:
            self.num_shapes = 0
            self.num_vertices = 0
        else:
            self.num_shapes = ((self.num_rows // 2 + 1) * (self.num_rows + 1) +
                               (self.num_rows // 2) * self.num_rows)
            self.num_vertices = (self.num_shapes *
                                 len(self.shape_vertices) *
                                 (1 if self.filled else 2))

    def set_vertices_and_codes(self, vertices, codes):
        offset = 1.0 / self.num_rows
        shape_vertices = self.shape_vertices * offset * self.size
        shape_codes = self.shape_codes
        if not self.filled:
            shape_vertices = np.concatenate(  # Forward, then backward.
                [shape_vertices, shape_vertices[::-1] * 0.9])
            shape_codes = np.concatenate([shape_codes, shape_codes])
        vertices_parts = []
        codes_parts = []
        for row in range(self.num_rows + 1):
            if row % 2 == 0:
                cols = np.linspace(0, 1, self.num_rows + 1)
            else:
                cols = np.linspace(offset / 2, 1 - offset / 2, self.num_rows)
            row_pos = row * offset
            for col_pos in cols:
                vertices_parts.append(shape_vertices + [col_pos, row_pos])
                codes_parts.append(shape_codes)
        np.concatenate(vertices_parts, out=vertices)
        np.concatenate(codes_parts, out=codes)


class Circles(Shapes):
    def __init__(self, hatch, density):
        path = Path.unit_circle()
        self.shape_vertices = path.vertices
        self.shape_codes = path.codes
        super().__init__(hatch, density)


class SmallCircles(Circles):
    size = 0.2

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('o')) * density
        super().__init__(hatch, density)


class LargeCircles(Circles):
    size = 0.35

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('O')) * density
        super().__init__(hatch, density)


class SmallFilledCircles(Circles):
    size = 0.1
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('.')) * density
        super().__init__(hatch, density)


class Stars(Shapes):
    size = 1.0 / 3.0
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = (hatch.count('*')) * density
        path = Path.unit_regular_star(5)
        self.shape_vertices = path.vertices
        self.shape_codes = np.full(len(self.shape_vertices), Path.LINETO,
                                   dtype=Path.code_type)
        self.shape_codes[0] = Path.MOVETO
        super().__init__(hatch, density)

_hatch_types = [
    HorizontalHatch,
    VerticalHatch,
    NorthEastHatch,
    SouthEastHatch,
    SmallCircles,
    LargeCircles,
    SmallFilledCircles,
    Stars
    ]


def _validate_hatch_pattern(hatch):
    valid_hatch_patterns = set(r'-+|/\xXoO.*')
    if hatch is not None:
        invalids = set(hatch).difference(valid_hatch_patterns)
        if invalids:
            valid = ''.join(sorted(valid_hatch_patterns))
            invalids = ''.join(sorted(invalids))
            _api.warn_deprecated(
                '3.4',
                removal='3.11',  # one release after custom hatches (#20690)
                message=f'hatch must consist of a string of "{valid}" or '
                        'None, but found the following invalid values '
                        f'"{invalids}". Passing invalid values is deprecated '
                        'since %(since)s and will become an error %(removal)s.'
            )


def get_path(hatchpattern, density=6):
    """
    Given a hatch specifier, *hatchpattern*, generates Path to render
    the hatch in a unit square.  *density* is the number of lines per
    unit square.
    """
    density = int(density)

    patterns = [hatch_type(hatchpattern, density)
                for hatch_type in _hatch_types]
    num_vertices = sum([pattern.num_vertices for pattern in patterns])

    if num_vertices == 0:
        return Path(np.empty((0, 2)))

    vertices = np.empty((num_vertices, 2))
    codes = np.empty(num_vertices, Path.code_type)

    cursor = 0
    for pattern in patterns:
        if pattern.num_vertices != 0:
            vertices_chunk = vertices[cursor:cursor + pattern.num_vertices]
            codes_chunk = codes[cursor:cursor + pattern.num_vertices]
            pattern.set_vertices_and_codes(vertices_chunk, codes_chunk)
            cursor += pattern.num_vertices

    return Path(vertices, codes)


attrs = {
    'scale': 6,
    'weight': 1.0,
    'angle': 0.0,
    'random_rotation': False,
    'random_placement': False,
    'x_stagger': 0.5,
    'y_stagger': 0.0,
    'filled': True,
}


class HatchStyle:
    def __init__(self, hatchpattern, **kwargs):
        self.hatchpattern = hatchpattern
        self.kwargs = {
            attr: kwargs.get(attr, default) for attr, default in attrs.items()
        }

    def rotate_vertices(self, vertices, angle=None, scale_correction=True):
        vertices = vertices.copy()

        if angle is None:
            angle = self.kwargs["angle"]
        angle_rad = np.deg2rad(angle)

        center = np.mean(vertices, axis=0)
        vertices -= center

        if scale_correction:
            scaling = abs(np.sin(angle_rad)) + abs(np.cos(angle_rad))
            vertices *= scaling

        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        vertices = np.dot(vertices, rotation_matrix)
        vertices += center
        return vertices

    def get_vertices_and_codes(self, hatch_buffer_size=100):
        self.hatch_buffer_size = hatch_buffer_size
        vertices, codes = np.empty((0, 2)), np.empty(0, Path.code_type)

        if self.hatchpattern in hatchpatterns:
            # This is for line hatches
            for func in np.atleast_1d(hatchpatterns[self.hatchpattern]):
                vertices_part, codes_part = func(self)
                vertices_part = self.rotate_vertices(vertices_part)

                vertices = np.concatenate((vertices, vertices_part))
                codes = np.concatenate((codes, codes_part))
        else:
            # This is for marker hatches
            if self.hatchpattern not in MarkerHatchStyle.marker_paths:
                raise ValueError(f"Unknown hatch pattern: {self.hatchpattern}")
            func = MarkerHatchStyle.marker_pattern
            vertices_part, codes_part = func(self)

            vertices = np.concatenate((vertices, vertices_part))
            codes = np.concatenate((codes, codes_part))

        return vertices, codes


class MarkerHatchStyle(HatchStyle):
    marker_paths = {
        'o': Path.unit_circle,
        'O': Path.unit_circle,
        '*': (Path.unit_regular_star, 5),  # TODO: is there a better way to do this?
    }

    # TODO: saner defaults or no?
    marker_sizes = {
        'o': 0.2,
        'O': 0.35,
        '*': 1.0 / 3.0,
    }

    def _get_marker_path(marker):
        func = np.atleast_1d(MarkerHatchStyle.marker_paths[marker])
        path = func[0](*func[1:])
        size = MarkerHatchStyle.marker_sizes.get(marker, attrs['weight'])

        return Path(
            vertices=path.vertices * size,
            codes=path.codes,
        )

    def marker_pattern(hatchstyle):
        size = hatchstyle.kwargs['weight']
        num_rows = round(
            hatchstyle.kwargs['scale'] * hatchstyle.hatch_buffer_size / 100.0
        )
        path = MarkerHatchStyle._get_marker_path(hatchstyle.hatchpattern)
        marker_vertices = hatchstyle.rotate_vertices(
            path.vertices, scale_correction=False
        )
        marker_codes = path.codes

        offset = 1.0 / num_rows
        marker_vertices = marker_vertices * offset * size
        x_stagger = hatchstyle.kwargs['x_stagger'] * offset
        y_stagger = hatchstyle.kwargs['y_stagger'] * offset

        if not hatchstyle.kwargs['filled']:
            marker_vertices = np.concatenate(
                [marker_vertices, marker_vertices[::-1] * 0.9]
            )
            marker_codes = np.concatenate([marker_codes, marker_codes])

        vertices = np.empty((0, 2))
        codes = np.empty(0, Path.code_type)
        for row in range(num_rows + 1):
            row_pos = row * offset
            if row % 2 == 0:
                cols = np.linspace(0, 1, num_rows + 1)
            else:
                cols = np.linspace(x_stagger, 1 + x_stagger, num_rows + 1)

            for i, col_pos in enumerate(cols):
                vertices_part = marker_vertices + [col_pos, row_pos]
                if i % 2 == 1:
                    vertices_part += [0, y_stagger]

                if hatchstyle.kwargs['random_rotation']:
                    vertices_part = hatchstyle.rotate_vertices(
                        vertices_part, np.random.uniform(0, 360), scale_correction=False
                    )

                if hatchstyle.kwargs['random_placement']:
                    vertices_part += np.random.uniform(-offset / 4, offset / 4, 2)

                vertices = np.concatenate((vertices, vertices_part))
                codes = np.concatenate((codes, marker_codes))

        return vertices, codes


class LineHatchStyle(HatchStyle):
    def horizontal(hatchstyle):
        num_lines = round(
            hatchstyle.kwargs['scale'] * hatchstyle.hatch_buffer_size / 100.0
        )
        if num_lines:
            num_vertices = num_lines * 2
        else:
            num_vertices = 0

        vertices = np.empty((num_vertices, 2))
        codes = np.empty(num_vertices, Path.code_type)
        steps, stepsize = np.linspace(0.0, 1.0, num_lines, False, retstep=True)
        steps += stepsize / 2.0
        vertices[0::2, 0] = 0.0
        vertices[0::2, 1] = steps
        vertices[1::2, 0] = 1.0
        vertices[1::2, 1] = steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

        return vertices, codes

    def vertical(hatchstyle):
        num_lines = round(
            hatchstyle.kwargs['scale'] * hatchstyle.hatch_buffer_size / 100.0
        )
        if num_lines:
            num_vertices = num_lines * 2
        else:
            num_vertices = 0

        vertices = np.empty((num_vertices, 2))
        codes = np.empty(num_vertices, Path.code_type)
        steps, stepsize = np.linspace(0.0, 1.0, num_lines, False, retstep=True)
        steps += stepsize / 2.0
        vertices[0::2, 0] = steps
        vertices[0::2, 1] = 0.0
        vertices[1::2, 0] = steps
        vertices[1::2, 1] = 1.0
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

        return vertices, codes

    def north_east(hatchstyle):
        num_lines = round(
            hatchstyle.kwargs['scale'] * hatchstyle.hatch_buffer_size / 100.0
        )
        if num_lines:
            num_vertices = (num_lines + 1) * 2
        else:
            num_vertices = 0

        vertices = np.empty((num_vertices, 2))
        codes = np.empty(num_vertices, Path.code_type)
        steps = np.linspace(-0.5, 0.5, num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

        return vertices, codes

    def south_east(hatchstyle):
        num_lines = round(
            hatchstyle.kwargs['scale'] * hatchstyle.hatch_buffer_size / 100.0
        )
        if num_lines:
            num_vertices = (num_lines + 1) * 2
        else:
            num_vertices = 0

        vertices = np.empty((num_vertices, 2))
        codes = np.empty(num_vertices, Path.code_type)
        steps = np.linspace(-0.5, 0.5, num_lines + 1)
        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 1.0 + steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 0.0 + steps
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO

        return vertices, codes


hatchpatterns = {
    '-': LineHatchStyle.horizontal,
    '|': LineHatchStyle.vertical,
    '/': LineHatchStyle.north_east,
    '\\': LineHatchStyle.south_east,
    '+': (LineHatchStyle.horizontal, LineHatchStyle.vertical),
    'x': (LineHatchStyle.north_east, LineHatchStyle.south_east),
}
