import typing as tp
from functools import cached_property
from collections import deque, Counter
from scipy.special import log_ndtr
from libs_new.utils import do_sth_to_each_row_of, recursively_convert_to_tuple, log_minus_exp, better_random_choice
import numpy as np
from itertools import cycle, islice
from libs_new.stack_manager import stack_run, StackElement, StackReturn, iterate_over_tree, TreeIterator

VERY_SMALL = 1e-12

class Dimensions:
    """
    represents a d-dimensional box.
    """
    def __init__(self, dims: np.ndarray):
        """
        :param dims: a numpy array of shape d x 2
        """
        assert dims.shape[1] == 2
        self._dims = dims

    @property
    def dims(self) -> np.ndarray:
        return self._dims.copy()

    @property
    def lows(self):
        return self.dims[:,0]

    @property
    def highs(self):
        return self.dims[:,1]

    def __contains__(self, item: np.ndarray) -> bool:
        # tested
        """
        Test whether a point is in the box
        :param item: a numpy array of length `d` (where `d` is the length of the box) representing a point in R^d.
        """
        sdim = do_sth_to_each_row_of(self.dims, item, '-')
        return np.all(sdim[:,0] <= VERY_SMALL) and np.all(sdim[:,1] >= -VERY_SMALL)

class UnsplittableError(Exception):
    pass

class Node:
    """
    A node in the tree of the mesh.
    """
    def __init__(self, left: 'Cell', right: 'Cell', decision_coordinate: int, decision_threshold: float, bearing_cell: 'Cell'):
        """
        Note: A newly created node always has a `Cell` on the left and a `Cell` on the right.
        """
        self.left: tp.Union[Cell, Node] = left
        self.right: tp.Union[Cell, Node] = right
        self.decision_coordinate = decision_coordinate
        self.decision_threshold = decision_threshold
        self.bearing_cell = bearing_cell

    def __repr__(self):
        return 'Dx{} < {}?'.format(self.decision_coordinate, self.decision_threshold)

class Cell:
    def __init__(self, content: tp.Union[tp.List[int], np.ndarray], dimensions: Dimensions, ref_table: np.ndarray):
        self.content = content
        self.dimensions = dimensions
        self.ref_table = ref_table

    def split(self, j:int) -> Node:
        # tested
        j = self._next_splittable_coordinate(j)
        if j is None:
            raise UnsplittableError
        partition = partitionner(self.ref_table[self.content][:,j])
        left_idx = self.content[partition.left_idx]
        right_idx = self.content[partition.right_idx]

        left_dimensions = self.dimensions.dims  # already copied
        assert left_dimensions[j, 1] >= partition.threshold
        left_dimensions[j, 1] = partition.threshold
        left_dimensions = Dimensions(left_dimensions)

        right_dimensions = self.dimensions.dims
        assert right_dimensions[j, 0] <= partition.threshold
        right_dimensions[j,0] = partition.threshold
        right_dimensions = Dimensions(right_dimensions)

        left_cell = Cell(content=left_idx, dimensions=left_dimensions, ref_table=self.ref_table)
        right_cell = Cell(content=right_idx, dimensions=right_dimensions, ref_table=self.ref_table)
        new_node = Node(left=left_cell, right=right_cell, decision_coordinate=j, decision_threshold=partition.threshold, bearing_cell=self)

        return new_node

    def _next_splittable_coordinate(self, j:int) -> tp.Union[None, int]:
        # tested
        real_content = self.ref_table[self.content]
        d = real_content.shape[1]
        for i,_ in zip(islice(cycle(range(d)), j, None, 1), range(d)):
            if _has_more_than_one_distinct_elem(real_content[:,i]):
                # if len(np.unique(real_content[:,i])) != 1
                return i
        return None

    def is_consistent(self) -> bool:
        for c in self.content:
            if self.ref_table[c] not in self.dimensions:
                return False
        return True

    def __hash__(self) -> int:
        content = recursively_convert_to_tuple(self.content)
        dimension = recursively_convert_to_tuple(self.dimensions.dims)
        return hash(content) + hash(dimension)

    def __repr__(self):
        return 'Content:{}\nDimensions:{}'.format(self.content, self.dimensions.dims.tolist())

def _has_more_than_one_distinct_elem(container: tp.Iterable):
    first = next(iter(container))
    for e in container:
        if e != first:
            return True
    return False

class Partition(tp.NamedTuple):
    left_idx: np.ndarray
    right_idx: np.ndarray
    threshold: float

def partitionner(x: tp.Sequence[float]) -> Partition:
    # tested
    N = len(x)
    k = int(N/2)
    x = np.array(x)
    argpart = np.argpartition(x, k)
    return Partition(left_idx=argpart[0:k], right_idx=argpart[k:], threshold=x[argpart[k]])

def _write_to_cell_map(cell: Cell, cell_map: tp.Dict[int, Cell]) -> None:
    # tested
    """
    Associate, in `cell_map`, every index that is contained in `cell`'s content to `cell`.
    """
    for i in cell.content:
        assert cell_map.get(i) is None
        cell_map[i] = cell

class BranchCellSplitter(StackElement):
    # tested
    def __init__(self, root: Node, node: Node, side: tp.Literal['left', 'right'], cell_map: tp.Dict[int, Cell], idx: int):
        """
        :param node: the node that parents the cell we want to split
        """
        self.root = root
        self.node = node
        self.side = side
        self.cell_map = cell_map
        self.idx = idx
        self.cell: Cell = getattr(self.node, self.side)
        self.d = self.cell.ref_table.shape[1]

    def stack(self) -> StackReturn:
        try:
            new_node = self.cell.split(self.idx)
        except UnsplittableError:
            _write_to_cell_map(self.cell, self.cell_map)
            return StackReturn(return_obj=self.root, new_elems=[])
        else:
            setattr(self.node, self.side, new_node)
            next_idx = (new_node.decision_coordinate + 1) % self.d
            # noinspection PyTypeChecker
            new_splitters = [BranchCellSplitter(root=self.root, node=new_node, side=s, cell_map=self.cell_map, idx=next_idx) for s in ('left', 'right')]
            return StackReturn(return_obj=self.root, new_elems=new_splitters)

class RootCellSplitter(StackElement):
    # tested
    def __init__(self, cell: Cell, cell_map: tp.Dict[int, Cell]):
        self.cell = cell
        self.cell_map = cell_map
        self.d = self.cell.ref_table.shape[1]

    def stack(self) -> StackReturn:
        try:
            new_node = self.cell.split(0)
        except UnsplittableError:
            _write_to_cell_map(self.cell, self.cell_map)
            return StackReturn(self.cell, [])
        else:
            next_idx = (new_node.decision_coordinate + 1) % self.d
            # noinspection PyTypeChecker
            new_splitters = [BranchCellSplitter(root=new_node, node=new_node, side=s, cell_map=self.cell_map, idx=next_idx) for s in ('left', 'right')]
            return StackReturn(return_obj=new_node, new_elems=new_splitters)

class Mesh:
    # tested
    """
    A Kd-tree-like container for points in R^d. Points are refered to by indices with respect to a given initial array.

    Known issue: if there are duplicate points or duplicate coordinates, then some of the boxes may be degenerated to sigletons. This may be fixed by changing the `partitionner` function in the mesh.py file.
    """
    def __init__(self, arr: np.ndarray):
        """
        :param arr: array of shape (N,d) representing N points in d dimensions.
        """
        self.ref_arr = arr
        self._cell_map: tp.Dict[int, Cell] = dict()  # map each index to its cell
        self._root = stack_run(self, self.__class__._stack_initialize)

    def _stack_initialize(self) -> tp.Deque[RootCellSplitter]:
        d = self.ref_arr.shape[1]
        starting_dims = Dimensions(np.array([[-np.inf, np.inf]] * d))
        starting_cell = Cell(content=np.arange(len(self.ref_arr)), dimensions=starting_dims, ref_table=self.ref_arr)
        return deque([RootCellSplitter(cell=starting_cell, cell_map=self._cell_map)])

    def get_cell_containing(self, x: np.ndarray) -> Cell:
        """
        :param x: numpy array of length `d`
        """
        res = self._root
        while not isinstance(res, Cell):
            assert isinstance(res, Node)
            if x[res.decision_coordinate] >= res.decision_threshold:
                res = res.right
            else:
                res = res.left
        return res

    def get_cell_via_idx(self, i:int) -> Cell:
        return self._cell_map[i]

    def iter_over_tree(self, method: tp.Literal['BFS', 'DFS']) -> tp.Iterable:
        return iterate_over_tree(self._root, self._get_children, method)

    @staticmethod
    def _get_children(elem) -> tp.Sequence[tp.Union[Cell, Node]]:
        assert isinstance(elem, Cell) or isinstance(elem, Node)
        if isinstance(elem, Node):
            return [elem.left, elem.right]
        else:
            return []

    @cached_property
    def cell_set(self):
        return set(self._cell_map.values())

    @staticmethod
    def _repr_in_tree(item: tp.Union[Node, Cell]):
        if isinstance(item, Node):
            return str(item)
        else:
            assert isinstance(item, Cell)
            return str(item.content.tolist())

    def print(self):
        """
        Beware. Result can be large.
        """
        res = ''
        old_depth = 0
        cell_dims = []
        for iterater in self.iter_over_tree('BFS'):
            iterater: TreeIterator
            if iterater.depth == old_depth:
                res += self._repr_in_tree(iterater.item) + '/'
            else:
                assert iterater.depth == old_depth + 1
                res += '\n' + self._repr_in_tree(iterater.item) + '/'
                old_depth += 1
            if isinstance(iterater.item, Cell):
                cell_dims.append(iterater.item.dimensions.dims.ravel())
        res += '\nDimensions:\n' + str(np.array(cell_dims))
        return res

def static_test_mesh(mesh: Mesh):
    """
    Verify that:
    (a) The cell map coincides with the set of all cells in the tree, called the cell set
    (b) All cells in the cell set are consistent
    (c) Each point in the reference table is in exactly one cell in the cell set, and that cell is given by the cell map
    """
    ref_table = mesh.ref_arr
    N = len(ref_table)
    # noinspection PyProtectedMember
    cell_map = mesh._cell_map
    cell_set = set(cell_map.values())

    cell_set_bis = []
    for tree_iterator in mesh.iter_over_tree('BFS'):
        tree_iterator: TreeIterator
        if isinstance(tree_iterator.item, Cell):
            cell_set_bis.append(tree_iterator.item)
    assert cell_set == set(cell_set_bis)

    assert all([c.is_consistent() for c in cell_set])
    assert Counter(cell_map.keys()) == Counter(range(N))

    all_contents = []
    for cell in cell_set:
        all_contents.extend(cell.content)
    assert Counter(all_contents) == Counter(range(N))

    for i, c in cell_map.items():
        assert i in c.content

def dynamic_test_mesh(mesh: Mesh, x: np.ndarray):
    """
    Test whether:
    (a) x belongs to exactly one cell in the cell set
    (b) get_cell_containing_x directs exactly to that cell

    :param x: A point in R^d that does not lay on cell borders
    """
    test = [(cell, (x in cell.dimensions)) for cell in mesh.cell_set]
    bools = [t[1] for t in test]
    assert sum(bools) == 1
    good_cell = test[bools.index(True)][0]
    assert mesh.get_cell_containing(x) is good_cell

class DiscreteGaussianDistribution:
    # tested
    """
    This discrete distribution is supported on the set {0, 1, ..., N-1}, where the actual points
    X_0, ..., X_{N-1} are contained in the reference table of the Mesh object.
    """

    def __init__(self, mesh: Mesh, means: tp.Sequence[float], variances: tp.Sequence[float]):
        self.mesh = mesh
        self.means = means
        self.variances = variances
        # noinspection PyTypeChecker
        self.scales: tp.Sequence[float] = self.variances ** 0.5
        if not (self.mesh.ref_arr.shape[1] == len(self.means) == len(self.variances)):
            raise ValueError

    def rvs(self):
        z = np.random.normal(loc=self.means, scale=self.scales)
        cell = self.mesh.get_cell_containing(z)
        assert len(cell.content) == 1  # We have not supported duplicated points yet.
        return cell.content[0]

    def logpdf(self, i: int) -> float:
        cell = self.mesh.get_cell_via_idx(i)
        dims = cell.dimensions.dims
        standardized_dims = do_sth_to_each_row_of(do_sth_to_each_row_of(dims, self.means, '-'), self.scales, '/')
        log_cdfs = log_ndtr(standardized_dims)
        # noinspection PyTypeChecker
        return sum(log_minus_exp(log_cdfs[:,1], log_cdfs[:,0]))

TreeElem = tp.Union[Cell, Node]

class NeighborMesh:
    # tested
    """
    A NeighborMesh is a partition of points into "neighboring sets".
    """
    def __init__(self, arr: np.ndarray, max_size: int):
        """
        Initialize a new NeighborMesh object
        :param arr: array of shape (N,d)
        :param max_size: maximum size of each neighboring set
        """
        self.arr = arr
        self.max_size = max_size

    @cached_property
    def _mesh(self) -> Mesh:
        return Mesh(self.arr)

    def _get_children(self, tree_elem: TreeElem) -> tp.Sequence[TreeElem]:
        assert isinstance(tree_elem, Cell) or isinstance(tree_elem, Node)
        if isinstance(tree_elem, Node) and len(tree_elem.bearing_cell.content) > self.max_size:
            return [tree_elem.left, tree_elem.right]
        return []

    @cached_property
    def _sets_of_neighbors(self) -> tp.Sequence[tp.Sequence[int]]:
        res = []
        # noinspection PyProtectedMember
        tree_iterator = iterate_over_tree(root=self._mesh._root, get_children=self._get_children, method='DFS')
        for i in tree_iterator:
            if i.is_leaf:
                if isinstance(i.item, Node):
                    res.append(i.item.bearing_cell.content)
                elif isinstance(i.item, Cell):
                    res.append(i.item.content)
                else:
                    raise AssertionError
        return res

    @cached_property
    def _set_map(self) -> tp.Dict[int, tp.Sequence[int]]:
        res = {}
        for neighbor_set in self._sets_of_neighbors:
            for p in neighbor_set:
                res[p] = neighbor_set
        return res

    def get_neighbor_set_of(self, i:int):
        return self._set_map[i]

    def rvs_markov(self, i: int):
        return better_random_choice(self.get_neighbor_set_of(i))