"""
Global scheme:
input -(stack initializer)-> stack -(stack emptier, applied recursively)-> output
"""

# tested

import typing as tp
from abc import ABC, abstractmethod
from collections import deque

RetTyp = tp.TypeVar('RetTyp')
InputTyp = tp.TypeVar('InputTyp')

class StackReturn(tp.NamedTuple):
    return_obj: RetTyp
    new_elems: tp.Sequence['StackElement']

class StackElement(ABC):
    @abstractmethod
    def stack(self) -> StackReturn:
        ...

def stack_run(input_obj: InputTyp, stack_initializer: tp.Callable[[InputTyp], tp.Deque[StackElement]]) -> RetTyp:
    stack = stack_initializer(input_obj)
    res = None
    while len(stack) > 0:
        stack_return = stack.pop().stack()
        res = stack_return.return_obj
        stack.extendleft(stack_return.new_elems)
    return res

class stack_iterator:
    def __init__(self, first_elem: StackElement, method: tp.Literal['BFS', 'DFS']):
        self._deque = deque([first_elem])
        self.method = method

    def extend(self, x: tp.Sequence):
        if self.method == 'BFS':
            return self._deque.extendleft(x)
        elif self.method == 'DFS':
            return self._deque.extend(reversed(x))
        else:
            raise ValueError('Unknown method')

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._deque) == 0:
            raise StopIteration
        else:
            stack_return = self._deque.pop().stack()
            self.extend(stack_return.new_elems)
            return stack_return.return_obj

TreeElem = tp.TypeVar('TreeElem')

class TreeIterator(StackElement):
    def __init__(self, root: TreeElem, item: TreeElem, get_children: tp.Callable[[TreeElem], tp.Sequence[TreeElem]], path: tp.List[int]):
        """
        Parameters
        ----------
        get_children: a function that takes a tree element (node/root/leaf, etc...) and returns its children
        path: path from the root to the `item`. For a binary tree, `path` will be a sequence of 0's and 1's
        """
        self.root = root
        self.get_children = get_children
        self.item = item
        self.path = path
        self.is_leaf = False

    def stack(self) -> StackReturn:
        new_elems = [self.__class__(root=self.root, item=c, get_children=self.get_children, path=self.path + [i]) for i, c in enumerate(self.get_children(self.item))]
        if len(new_elems) == 0:
            self.is_leaf = True
        return StackReturn(self, new_elems)

    @property
    def depth(self) -> int:
        return len(self.path)

def iterate_over_tree(root: TreeElem, get_children: tp.Callable[[TreeElem], tp.Sequence[TreeElem]], method: tp.Literal['BFS', 'DFS']) -> tp.Union[stack_iterator, tp.Iterable[TreeIterator]]:
    # tested
    """
    Returns an iterator over a tree. Both breadth-first methods and depth-first methods are supported.

    Parameters
    ----------

    root: root of the tree
    get_children: callable that takes a tree's root, leaf or node and returns its children in the expected order. The function may return an empty list.

    Returns
    -------

    An iterator over the tree. Each element of the iterator has an `item` and a `path` attribute.

    """
    first_elem = TreeIterator(root=root, item=root, get_children=get_children, path=[])
    return stack_iterator(first_elem, method)

if __name__ == '__main__':
    # Example 1:
    # Let us build a word tree as an example
    class WordNode:
        def __init__(self, content: str, left: tp.Union['WordNode', None], right: tp.Union['WordNode', None]):
            self.content = content
            self.left = left
            self.right = right

        def __repr__(self):
            return self.content

        def split(self) -> tp.Union[tp.Tuple['WordNode', 'WordNode'], None]:
            if len(self.content) <= 1:
                return None
            else:
                N = len(self.content)
                k = int(N/2)
                return self.__class__(self.content[0:k], None, None), self.__class__(self.content[k:], None, None)

    # As you have seen, the `split` method does not modify the node.
    # List of actions to do are instead managed by a list (called a stack) whose elements are of type `StackElement`.
    # Each `StackElement` (again, a.k.a thing-to-do) generates two more things-to-do.

    class NodeSeparator(StackElement):
        def __init__(self, root: WordNode, node_to_split: WordNode):
            self.root = root
            self.node_to_split = node_to_split

        def stack(self) -> StackReturn:
            two_new_nodes = self.node_to_split.split()
            if two_new_nodes is None:
                return StackReturn(self.root, [])
            else:
                left, right = two_new_nodes
                self.node_to_split.left = left
                self.node_to_split.right = right
                return StackReturn(self.root, [self.__class__(self.root, left), self.__class__(self.root, right)])

    def word_tree_initializer(word: str) -> tp.Deque[NodeSeparator]:
        root = WordNode(word, None, None)
        return deque([NodeSeparator(root, root)])

    # Let's get started.
    my_tree = stack_run('Hello_world', word_tree_initializer)
    current_level = [my_tree]
    while len(current_level) > 0:
        new_level = []
        for n in current_level:
            if n.left is None:
                assert n.right is None
            else:
                new_level.append(n.left)
                new_level.append(n.right)
        print(''.join(['{}/'.format(n) for n in current_level]))
        current_level = new_level

