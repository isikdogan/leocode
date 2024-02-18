LL90: Leo's LeetCode 90
---

## Two Sum
[link]https://leetcode.com/problems/two-sum[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def twoSum(self, nums, target):
        # Initialize an empty dictionary to store elements of the list as keys and their indices as values
        res = {}
        
        # Loop through the list "nums" along with their indices
        for i, n in enumerate(nums):
            # Calculate the complement needed to reach the target sum
            complement = target - n
            
            # Check if the complement is already in the dictionary "res"
            if complement in res:
                # If found, return the index of the complement and the current index
                return [res[complement], i]
            
            # If not, add the current element to the dictionary with its index as the value
            res[n] = i
</code></pre>

* Time complexity: O(n) for traversing the list once
* Space complexity: O(n) for the dictionary.

^^^

## Contains Duplicate
[link]https://leetcode.com/problems/contains-duplicate[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def containsDuplicate(self, nums):

        # Create an empty set to keep track of unique elements
        visited = set()

        for n in nums:
            # If already seen, then we've found a duplicate
            if n in visited:
                return True
            visited.add(n)

        return False
</code></pre>

* Time and space complexity: O(n)

^^^

## Majority Element
[link]https://leetcode.com/problems/majority-element[/link]
[tag]Easy[/tag]

<pre><code class="language-python">from collections import Counter

class Solution:
    """ Solution 1: Use Python's built-in Counter to count occurrences. """
    def majorityElement(self, nums):
        # Count occurrences of each element (same as counter[num] = occurence)
        counter = Counter(nums)

        # Loop through the counter to find the majority element
        # i.e. its count is greater than half the size of the list
        for n, count in counter.items():
            if count > len(nums) // 2:
                return n
</code></pre>

* Time complexity: O(n) for the loop.  
* Space complexity: O(n) for the counter.

---

<pre><code class="language-python">class Solution:
    """ Solution 2: Use Boyer-Moore Voting Algorithm. """
    def majorityElement(self, nums):
        # This will hold the potential majority element.
        cur = nums[0]
        
        # This will help us determine if 'cur' is indeed the majority element.
        counter = 1

        # Iterate through the rest of the elements in the list.
        # If the current element 'n' is the same as 'cur', increment the counter.
        # If it's different, then decrement the counter.
        # If the counter reaches 0, assign 'n' as the new candidate and reset the counter.
        for n in nums[1:]:
            if n == cur:
                counter += 1
            else:
                counter -= 1

                if counter == 0:
                    cur = n
                    counter = 1

        # We assume that a majority element always exists as per the problem's constraint.
        return cur
</code></pre>

* Time complexity: O(n).  
* Space complexity: O(1) since we use a constant number of variables.

^^^

## Missing Number
[link]https://leetcode.com/problems/missing-number[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def missingNumber(self, nums):
        # Expected sum of numbers - actual sum of numbers
        n = len(nums)
        return (n * (n + 1)) // 2 - sum(nums)
</code></pre>

* Time complexity: O(n)  
* Space complexity: O(1)

^^^

## Valid Anagram
[link]https://leetcode.com/problems/valid-anagram[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def isAnagram(self, s, t):
        # This is equivalent to Counter(s) == Counter(t)

        # Anagrams have the same length
        if len(s) != len(t):
            return False

        # Count each character in string 's'
        letters = {}
        for c in s:
            letters[c] = 1 if c not in letters else letters[c] + 1

        # Loop through each character in string 't'
        for c in t:
            if c not in letters or letters[c] == 0:
                return False
            else:
                letters[c] -= 1

        # If the loop completes without returning False, the strings are anagrams
        return True
</code></pre>

* Time and space complexity: O(n), where n is the length of the same length input strings

^^^

## Ransom Note
[link]https://leetcode.com/problems/ransom-note[/link]
[tag]Easy[/tag]

<pre><code class="language-python">from collections import Counter

class Solution:
    def canConstruct(self, ransomNote, magazine):
        # Use Counter to get the frequency of each character in the magazine.
        magazine_counter = Counter(magazine)
        
        # Check if each character in ransomNote can be constructed from the magazine.
        for c in ransomNote:
            if magazine_counter[c] == 0:
                return False
            magazine_counter[c] -= 1

        return True
</code></pre>

* Time complexity: O(m+n), where m and n are the lengths of magazine and ransomNote  
* Space complexity is O(m)

^^^

## Valid Parentheses
[link]https://leetcode.com/problems/valid-parentheses[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def isValid(self, s):
        # Use a stack to keep track of the opening parentheses.
        stack = []
        
        # Map closing parentheses to their corresponding opening parentheses
        parentheses = {")": "(", "}": "{", "]": "["}

        for c in s:
            # If "c" is an opening parenthesis, append it to the "stack".
            if c in parentheses.values():
                stack.append(c)
            # If "c" is a closing parenthesis then it should match with the top of stack
            elif not stack or stack.pop() != parentheses[c]:
                return False
        
        # If stack is empty then all opening parentheses are closed; otherwise, return False
        return not stack
</code></pre>

* Time complexity: O(n) where n is len(s), since each character in s is processed once, either pushed onto the stack or popped from the stack.
* Space complexity: O(n) in the worst case when all characters are opening parentheses, requiring a stack of the same size as the string s.

^^^

## Implement Queue using Stacks
[link]https://leetcode.com/problems/implement-queue-using-stacks[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class MyQueue:
    def __init__(self):
        # Initialize two stacks
        self.input_stack = []
        self.output_stack = []

    def push(self, x):
        # Push element onto input stack
        self.input_stack.append(x)

    def pop(self):
        # Move elements to output stack if it's empty
        self.shift_stacks()
        # Pop element from output stack
        return self.output_stack.pop()

    def peek(self):
        # Move elements to output stack if it's empty
        self.shift_stacks()
        # Return the top element of output stack
        return self.output_stack[-1]

    def empty(self):
        # Queue is empty if both stacks are empty
        return len(self.input_stack) == 0 and len(self.output_stack) == 0

    def shift_stacks(self):
        # Move elements from input stack to output stack if output stack is empty
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
</code></pre>

* Time complexity: O(1) for push and empty. Amortized O(1) for pop and peek: shifting elements from input_stack to output_stack takes O(n) when output_stack is empty, but each element is moved exactly once. Thus, the average time per operation is still constant.
* Space complexity: O(n) where n is the number of elements in the queue.

^^^

## Min Stack
[link]https://leetcode.com/problems/min-stack[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class MinStack:
    def __init__(self):
        # Initialize an empty list to act as the stack.
        # Each element in the stack will be a tuple (x, current_min),
        # where x is the value and current_min is the minimum value in the stack up to that point.
        self.stack = []

    def push(self, val):
        # If the stack is empty, the min value is the value itself.
        # Otherwise, the min value is the minimum between the current value and the current minimum.
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self):
        # Pop the top element from the stack, which removes the last tuple.
        self.stack.pop()

    def top(self):
        # Return the top element value (first element of the tuple).
        return self.stack[-1][0]

    def getMin(self):
        # Return the current minimum value (second element of the tuple).
        return self.stack[-1][1]
</code></pre>

* Time complexity: O(1) for all methods.
* Space complexity: O(n), where n is the number of elements in the stack. Each element in the stack is stored as a pair (value and current minimum), which increases the space requirement, but it's still linear in the size of the stack.

^^^

## Binary Search
[link]https://leetcode.com/problems/binary-search[/link]
[tag]Easy[/tag]

[Binary seach](https://en.wikipedia.org/wiki/Binary_search_algorithm) is a classic search algorithm that is used very frequently, both in interview questions and in real applications.
<pre><code class="language-python">class Solution:
    def search(self, nums, target):

        low = 0
        high = len(nums) - 1

        while low <= high:
            mid = (low + high) // 2

            if target == nums[mid]:
                return mid
            elif target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        
        return -1
</code></pre>

* Time complexity: O(log n), where n is the number of elements in the array. Because with each comparison, it effectively halves the search space.
* Space complexity: O(1), as it only uses a constant amount of extra space for the pointers and the mid calculation.

^^^

## Reverse Linked List
[link]https://leetcode.com/problems/reverse-linked-list[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def reverseList(self, head):

        # This will eventually become the new head of the reversed list.
        prev = None

        # Traverse the list until we reach the end (when head becomes None).
        while head:
            # Set the next of the current node to the previous node (reversing the link).
            # Update prev to the current node (moving prev one step forward).
            # Move the head to the next node in the original list (head.next before reversal).
            head.next, prev, head = prev, head, head.next

        return prev
</code></pre>

* Time complexity: O(n), each node is visited exactly once.
* Space complexity: O(1), as the reversal is done in place.

^^^

## Merge Two Sorted Lists
[link]https://leetcode.com/problems/merge-two-sorted-lists[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def mergeTwoLists(self, list1, list2):

        # Initialize current node to a dummy head
        merged_head = ListNode()
        current = merged_head

        # Loop through both lists until one of them is exhausted.
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # After the main loop, at least one of the lists is exhausted.
        # Append the remaining part of the other list to the merged list.
        current.next = list2 if not list1 else list1

        # Return the head of the new list, which is next to the dummy node.
        return merged_head.next
</code></pre>

* Time complexity: O(n + m), where n is the length of list1 and m is the length of list2.
* Space complexity: O(1), as only a fixed number of extra pointers are used.

^^^

## Remove Nth Node From End of List
[link]https://leetcode.com/problems/remove-nth-node-from-end-of-list[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def removeNthFromEnd(self, head, n):

        p1 = p2 = head

        # Advance the first pointer (p1) n steps ahead in the list.
        for _ in range(n):
            p1 = p1.next

        # If p1 is None after moving n steps, the node to remove is the head.
        # Hence, return the second node as the new head of the list.
        if not p1:
            return head.next
        
        # Move both pointers until p1 reaches the end of the list.
        # This will place p2 just before the node to be removed.
        while p1.next:
            p1 = p1.next
            p2 = p2.next

        # Remove the nth node from the end by skipping it.
        p2.next = p2.next.next

        return head
</code></pre>

* Time complexity: O(L), where L is the length of the linked list.
* Space complexity: O(1), only a fixed amount of extra space is used for pointers.

^^^

## Linked List Cycle
[link]https://leetcode.com/problems/linked-list-cycle[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def hasCycle(self, head):

        # Initialize two pointers, slow and fast
        slow = head
        fast = head

        # Loop until fast or fast.next becomes None
        while fast and fast.next:
            slow = slow.next          # Move slow pointer by 1 step
            fast = fast.next.next     # Move fast pointer by 2 steps

            # If slow and fast meet, there's a cycle
            if slow == fast:
                return True

        # If we exit the loop, there's no cycle
        return False
</code></pre>

* Time complexity: O(n). In the worst case, we might loop through all nodes once (if there's no cycle).
* Space complexity: O(1). We only use two pointers, slow and fast, regardless of the size of the linked list.

^^^

## Add Two Numbers
[link]https://leetcode.com/problems/add-two-numbers[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def addTwoNumbers(self, l1, l2):

        # Initialize current node to a dummy head and carry to zero.
        cur = head = ListNode()
        c = 0

        # Continue looping until both l1 and l2 are exhausted and there is no carry left.
        while l1 or l2 or c > 0:
            s = c
            if l1:
                s += l1.val
                l1 = l1.next
            if l2:
                s += l2.val
                l2 = l2.next
            c = s // 10

            cur.next = ListNode(s % 10)
            cur = cur.next
        
        # Return the head of the new list, which is next to the dummy node.
        return head.next
</code></pre>

* Time complexity: O(max(n, m)), where n and m are the lengths of l1 and l2.
* Space complexity: O(max(n, m)). The length of the new list is at most max(n, m) + 1 (due to a possible carry at the end).

^^^

## 3Sum
[link]https://leetcode.com/problems/3sum[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def threeSum(self, nums):
        # Sort the array for two-pointer approach
        nums.sort()
        n = len(nums)

        # Use a set to avoid duplicates
        result = set()

        for i in range(n - 2):
            left = i + 1
            right = n - 1

            # Use two-pointer technique to find the triplets
            while left < right:
                total = nums[i] + nums[left] + nums[right]

                # Check if the sum of the triplet is zero
                if total == 0:
                    # Add the triplet to the result set
                    result.add((nums[i], nums[left], nums[right]))
                    # Move both pointers
                    left += 1
                    right -= 1

                # If sum is less than zero, move the left pointer to the right
                elif total < 0:
                    left += 1
                # If sum is more than zero, move the right pointer to the left
                else:
                    right -= 1

        return list(result)
</code></pre>


* Time complexity: O(n^2). Sorting is O(n log n), outer loop O(n), inner _while_ loop up to O(n). Therefore, the total complexity is O(n * n + n log n) ≡ O(n^2).
* Space complexity: O(n), for the results.

^^^

## Valid Palindrome
[link]https://leetcode.com/problems/valid-palindrome[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def isPalindrome(self, s):
        left, right = 0, len(s) - 1

        # If a character at left or right is not alphanumeric, skip it.
        # Otherwise check if they are the same. If same, keep going.
        # If not, then it's not a palindrome.
        while left < right:
            if not s[left].isalnum():
                left += 1
            elif not s[right].isalnum():
                right -= 1
            elif s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1
        return True
</code></pre>

* Time complexity: O(n), where n is the length of the string s. Each character in the string is visited at most once.
* Space complexity: O(1), as no extra space is used that scales with the input size. Only two pointers are used regardless of the string's length.

^^^

## Longest Palindrome
[link]https://leetcode.com/problems/longest-palindrome[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def longestPalindrome(self, s):

        # Use a dictionary to count occurrences of each character.
        # This is the same as counter = Counter(s)
        counter = {}
        for c in s:
            counter[c] = 1 if c not in counter else counter[c] + 1

        # Round down occurences to the closest even number
        # because a palindrome requires pairs of characters.
        max_len = 0
        for c in counter:
            max_len += 2 * (counter[c] // 2)

        # Add 1 to max_len if there is a central character.
        if max_len == len(s):
            return max_len
        else:
            return max_len + 1
</code></pre>

* Time complexity: O(n), where n is the length of the string s.
* Space complexity: O(1), the space used by counter is constant (number of possible characters) regardless of the length of s.

^^^

## Valid Sudoku
[link]https://leetcode.com/problems/valid-sudoku[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def isValidSudoku(self, board):
        # Initialize three sets to track the seen numbers in rows, columns, and boxes
        seen_rows = [set() for _ in range(9)]
        seen_columns = [set() for _ in range(9)]
        seen_boxes = [set() for _ in range(9)]

        for i in range(9):  # Loop through rows
            for j in range(9):  # Loop through columns
                num = board[i][j]
                if num != '.':
                    # Calculate box index for the 3x3 sub-boxes
                    box_index = (i // 3) * 3 + j // 3

                    # Check if the number has already been seen in the current row, column, or box
                    if num in seen_rows[i] or num in seen_columns[j] or num in seen_boxes[box_index]:
                        return False  # Invalid Sudoku

                    # Add the number to the respective sets
                    seen_rows[i].add(num)
                    seen_columns[j].add(num)
                    seen_boxes[box_index].add(num)

        return True  # Sudoku is valid
</code></pre>

* Time and space complexity: O(1), because the size of the Sudoku board is constant (9x9).  If it wasn't fixed, then they would both be O(n^2), where n is the size of one dimension.

^^^

## Group Anagrams
[link]https://leetcode.com/problems/group-anagrams[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs):
        # Create a defaultdict with list as the default factory
        # If a key doesn't exist, this initializes it with an empty list
        anagrams = defaultdict(list)

        # Iterate through each string in the input list
        for s in strs:
            # Sort the string and use it as a key
            key = ''.join(sorted(s))
            
            # Append the original string to the list corresponding to the sorted key
            anagrams[key].append(s)

        # Return the values of the dictionary as a list of lists
        return list(anagrams.values())
</code></pre>

* Time complexity: O(N * K log K), N for the number of strings and _K log K_ for sorting the strings where K is the maximum length of a string.
* Space complexity: O(N * K), considering the space required for the output and the hash map.

^^^

## Counting Bits
[link]https://leetcode.com/problems/counting-bits[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def countBits(self, n):
        # This holds the count of 1's in binary for each number from 0 to n, inclusive.
        ans = [0] * (n + 1)

        # For each number, add the least significant bit of the current number (i % 2)
        # to the count of 1 bits in the right bit-shifted version of the number (i >> 1).
        # The bit-shifted number is always smaller, so its bit count is always precomputed.
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + i % 2
            
        return ans
</code></pre>

* Time complexity: O(n), where n is the input number. 
* Space complexity: O(n), as we are creating an array 'ans' of size n + 1 to store the results.

^^^

## Single Number
[link]https://leetcode.com/problems/single-number/[/link]
[tag]Easy[/tag]

This solution looks mind-boggling at first, but bear with me. It's a very elegant solution. To solve this problem efficiently, we can use the XOR operation. XOR is a bitwise operator that returns 0 if both bits are the same, and 1 otherwise. A useful property of XOR is that it cancels out identical numbers (i.e., a XOR a = 0). If you XOR a number with itself, the result is 0, and if you XOR a number with 0, the result is the original number. Since every number in the array appears twice except for one, if we XOR all the numbers together, the result will be the unique number.

<pre><code class="language-python">class Solution:
    def singleNumber(self, nums):
        unique = 0

        for num in nums:
            # Apply XOR between the current number and the result so far
            # This will cancel out numbers that appear twice
            unique ^= num

        # After processing all numbers, 'unique' will hold the single number
        return unique
</code></pre>

* Time complexity: O(n). We through the array once and each iteration involves a constant time XOR operation.
* Space complexity: O(1). It only uses a single integer (unique) to store the result.

Fun fact: We can use these properties of XOR to swap the values of two variables without needing an extra buffer. Here's how it's done: `a = a ^ b`, `b = a ^ b`, `a = a ^ b`. It's a neat idea! However, unless you're working on hardware-level optimizations, swapping variables like this is usually not a great idea. `a, b = b, a` is easier to read and works just fine.

^^^

## Climbing Stairs
[link]https://leetcode.com/problems/climbing-stairs[/link]
[tag]Easy[/tag]

<pre><code class="language-python"># This problem is very similar to Fibonacci series

class Solution:
    def climbStairs(self, n):
        # Base cases: 
        # There's one way to climb 0 stairs (do nothing) and 1 stair (single step).
        n_0, n_1 = 1, 1

        # The number of ways to climb the current stair is the sum
        # of the ways to climb the previous two sets of stairs,
        # since we can either climb 1 or 2 steps at a time.
        for i in range(n):
            n_0, n_1 = n_1, n_0 + n_1

        return n_0
</code></pre>

* Time complexity: O(n), where n is the number of stairs.
* Space complexity: O(1), since we use a constant number of variables.

^^^

## Product of Array Except Self
[link]https://leetcode.com/problems/product-of-array-except-self[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def productExceptSelf(self, nums):
        result = [1] * len(nums)

        # pre_prod and post_prod keep track of the product of elements
        # before and after the current element.
        pre_prod, post_prod = 1, 1

        for i in range(len(nums)):
            result[i] *= pre_prod
            result[-1-i] *= post_prod
            pre_prod *= nums[i]
            post_prod *= nums[-1-i]

        return result
</code></pre>

* Time complexity: O(n), where n is the length of the input.
* Space complexity: O(n), for the results array. O(1) excluding the output.

^^^

## Search in Rotated Sorted Array
[link]https://leetcode.com/problems/search-in-rotated-sorted-array[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    # We first determine which half of the array is properly sorted.
    # Then, we check if the target is within the sorted half.
    # If it is, the search continues in that half; otherwise, it switches to the other half.
    def search(self, nums, target):

        low, high = 0, len(nums) - 1
        
        while low <= high:
            mid = (low + high) // 2

            if target == nums[mid]:
                return mid

            # Left half is sorted.
            if nums[low] <= nums[mid]:
                if nums[low] <= target < nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            # Right half is sorted.
            else:
                if nums[mid] < target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        return -1
</code></pre>

* Time complexity: O(log n). Same as regular binary search.
* Space complexity is O(1), as the algorithm uses a constant amount of space, with variables low, high, and mid.

^^^

## Rotate Array
[link]https://leetcode.com/problems/rotate-array/[/link]
[tag]Medium[/tag]

Fun fact: I use this type of array rotation in many types projects, from computational art to computer vision, but I never needed to implement it from scratch, since _numpy.roll_ does the job.

<pre><code class="language-python">class Solution:
    """ Solution 1: copy array. """
    def rotate(self, nums: List[int], k: int) -> None:
        nums_orig = nums[:]
        for i in range(len(nums)):
            nums[i] = nums_orig[(i - k) % len(nums)]
</code></pre>
* Time complexity: O(n). Copying the array nums_orig = nums[:] takes O(n) time. The for loop also runs for N iterations. O(2n) = O(n).
* Space complexity: O(n), for the extra array nums_orig.

---

<pre><code class="language-python">class Solution:
    """ Solution 2: reverse the array in parts to achieve the rotation. """
    def rotate(self, nums, k):

        def reverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        n = len(nums)
        k %= n

        # Reverse the entire array
        reverse(0, n - 1)
        # Reverse the first k elements
        reverse(0, k - 1)
        # Reverse the remaining elements
        reverse(k, n - 1)
</code></pre>

* Time complexity: O(n). Each element is accessed a constant number of times (up to 3 times).
* Space complexity: O(1). The reversal is done in-place, and no additional space is used except for a few variables for iteration and indexing.

^^^

## Invert Binary Tree
[link]https://leetcode.com/problems/invert-binary-tree[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    # Recursively invert the left and right subtrees and swap them
    def invertTree(self, root):
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)

        return root
</code></pre>

* Time complexity: O(n), where n is the number of nodes in the tree. It visits each node once to swap its children.
* Space complexity: O(h), where h is the height of the tree. In the best-case scenario (a balanced tree), the space complexity would be O(log n), where log n is the height of a balanced tree.

^^^

## Maximum Depth of Binary Tree
[link]https://leetcode.com/problems/maximum-depth-of-binary-tree[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def maxDepth(self, root):

        if root is None:
            return 0

        # Return the maximum of left_depth and right_depth, plus 1 for the current node
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
</code></pre>

* Time complexity: O(n), where n is the number of nodes in the tree. This is because the algorithm must visit each node once to compute its depth.
* Space complexity: O(h), where h is the height of the tree. This accounts for the space used in the call stack during the recursive calls. In the worst case (a completely unbalanced tree), the space complexity can be O(n). However, in a balanced tree, it would be O(log n).

^^^

## House Robber
[link]https://leetcode.com/problems/house-robber[/link]
[tag]Medium[/tag]

<pre><code class="language-python"># This problem can also be solved with dynamic programming
# but I find using recursion with memoization more intuitive.

class Solution:
    def rob(self, nums):

        def rob_from_idx(i, nums, memo):
            # Base case: if index is beyond the last house
            if i >= len(nums):
                return 0

            # If this subproblem is already solved, return the stored result
            if i in memo:
                return memo[i]

            # Rob the current house and move to the house two steps ahead
            # or skip the current house and move to the next house
            rob_current = nums[i] + rob_from_idx(i + 2, nums, memo)
            skip_current = rob_from_idx(i + 1, nums, memo)

            # Store the maximum of the two options in memo
            memo[i] = max(rob_current, skip_current)

            return memo[i]

        # Start the recursion from the first house
        return rob_from_idx(0, nums, {})
</code></pre>

* Time complexity: O(n), where n is the number of houses. Without memoization it would have been exponential, i.e. O(2^n), because of the branching options at every call.
* Space complexity: O(n). It would still be O(n) without memoization because it does not depend on the number of total calls made, but on the depth of the call stack

^^^

## Task Scheduler
[link]https://leetcode.com/problems/task-scheduler[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import Counter

class Solution:
    def leastInterval(self, tasks, n):
        # Count frequencies of tasks
        counts = Counter(tasks)

        # Find the maximum frequency of any task
        max_freq = max(counts.values())

        # Count how many tasks have the maximum frequency
        num_max_tasks = 0
        for t in counts:
            if counts[t] == max_freq:
                num_max_tasks += 1

        # Calculate the number of intervals based on the max frequency
        intervals = max_freq - 1

        # Calculate the number of empty slots within the intervals
        # Adjusted by the number of tasks having the max frequency
        empty = intervals * (n - (num_max_tasks - 1))

        # Calculate the number of remaining tasks after placing the most frequent ones
        remaining = len(tasks) - num_max_tasks * max_freq

        # The total time is the sum of all tasks at max frequency and the larger of
        # the number of empty slots or remaining tasks
        return max_freq * num_max_tasks + max(empty, remaining)
</code></pre>

* Time complexity: O(n), where n is the number of tasks. We iterate over the tasks once to count their frequencies and then iterate over the count dictionary.
* Space complexity: O(m), where m is the number of unique tasks, for storing the frequency of each unique task.


^^^

## Gas Station
[link]https://leetcode.com/problems/gas-station[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def canCompleteCircuit(self, gas, cost):

        if sum(gas) < sum(cost): # No solution exists
            return -1

        # Iterate through each gas station and update the amount of gas in the tank
        start, tank = 0, 0
        for i in range(len(gas) - 1):
            tank += gas[i] - cost[i]

            # If we can't reach the next station,
            # move the starting station and reset the tank
            if tank < 0:
                start = i + 1
                tank = 0

        return start
</code></pre>

* Time complexity: O(n), where n is the number of gas stations. We iterate through the list of gas stations exactly once.
* Space complexity: O(1), as we are only using a constant amount of extra space for the variables.

^^^

## Coin Change
[link]https://leetcode.com/problems/coin-change[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def coinChange(self, coins, amount):
        # Create a dynamic programming array of length (amount + 1).
        # This value acts as a placeholder for the minimum coins needed for each amount.
        # It's initialized to amount + 1 because it's the upper bound of the minimum coins possible.
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0  # Base case: no coins needed for amount 0

        # Iterate through each type of coin available.
        # For each coin, we either use it (dp[x - coin] + 1) or not use it
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)

        # If dp[amount] is still amount+1, then return -1 as it's impossible to form the amount
        return -1 if dp[amount] == amount + 1 else dp[amount]
</code></pre>

* Time complexity: O(S * n), where S is the amount and n is the number of coins. For each coin, we iterate through all amounts from the coin's value up to the amount.
* Space complexity: O(S), for the dp array of size S+1.

^^^

## Diameter of Binary Tree
[link]https://leetcode.com/problems/diameter-of-binary-tree[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def diameterOfBinaryTree(self, root):
        self.diameter = 0

        def depth(node):
            if not node:
                return 0
            # Recursively find the height of the left and right subtrees
            left = depth(node.left)
            right = depth(node.right)

            # Update the diameter if the path through the current node is longer
            self.diameter = max(self.diameter, left + right)

            # Return the height of the current node
            return max(left, right) + 1

        depth(root)
        return self.diameter
</code></pre>

* Time complexity: O(N), where N is the number of nodes in the tree. We visit each node exactly once.
* The space complexity: O(H), where H is the height of the tree. This is due to the recursion stack during the DFS. In the worst case (a skewed tree), the space complexity can be O(N).

^^^

## Balanced Binary Tree
[link]https://leetcode.com/problems/balanced-binary-tree[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def isBalanced(self, root):
        def depth(node):
            if not node:
                return 0

            # Check the height of the left and right subtrees
            left = depth(node.left)
            right = depth(node.right)

            # If left or right subtree is unbalanced, propagate the failure
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1

            # Return the height of the current node
            return max(left, right) + 1

        # Check if the tree is balanced starting from the root
        return depth(root) != -1
</code></pre>

* Time complexity: O(n), where n is the number of nodes in the tree. Each node is visited once.
* Space complexity: O(h), where h is the height of the tree. This space is used by the call stack during recursion. In the worst case (a skewed tree), this could be O(n).

^^^

## Validate Binary Search Tree
[link]https://leetcode.com/problems/validate-binary-search-tree[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def isValidBST(self, root):

        def is_valid(node, lower, upper):

            if not node:
                return True

            # Check if node's value falls within the valid range
            if node.val >= upper or node.val <= lower:
                return False

            # Recursively validate the left and right subtree
            return is_valid(node.left, lower, node.val) and is_valid(node.right, node.val, upper)
        
        return is_valid(root, -float('inf'), float('inf'))
</code></pre>

* Time complexity: O(n), where n is the number of nodes in the tree. Each node is visited once.
* Space complexity: O(h), where h is the height of the tree. This is due to the recursion stack. In the worst case of a skewed tree, this can become O(n).

^^^

## Convert Sorted Array to Binary Search Tree
[link]https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def sortedArrayToBST(self, nums):
        if not nums:
            return None

        # Find the middle element of the array
        mid = len(nums) // 2

        # The middle element becomes the root of the BST
        root = TreeNode(nums[mid])

        # Recursively build the left subtree using the left half of the array
        root.left = self.sortedArrayToBST(nums[:mid])

        # Recursively build the right subtree using the right half of the array
        root.right = self.sortedArrayToBST(nums[mid + 1:])

        return root
</code></pre>

* Time complexity: O(N). Each element in the array is visited once to create a corresponding node.
* Space complexity: O(log N). This space is used by the recursion stack. The height of the BST will be log N for a height-balanced tree, so the maximum number of recursive calls on the stack at any time will be log N.

^^^

## Binary Tree Level Order Traversal
[link]https://leetcode.com/problems/binary-tree-level-order-traversal[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root:
            return []  # Return empty list if the tree is empty

        # Initialize a queue for Breadth-First Search (BFS).
        q = deque([root])

        # This will store the final level order traversal
        level_order = []

        # Continue until there are no more nodes to process
        while q:
            # Temporary list to store nodes at the current level
            level_nodes = []

            # Iterate over all nodes at the current level
            for _ in range(len(q)):
                # Pop the leftmost node
                node = q.popleft()

                # Add its value to the current level's list
                level_nodes.append(node.val)

                # Add the children of the current node to the queue
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            # Add the current level's nodes to the level order list
            level_order.append(level_nodes)

        return level_order
</code></pre>

* Time complexity: O(N), where N is the number of nodes in the tree. Each node is visited once.
* Space complexity: O(N), as we need space for the queue. In the worst case, we might need to store all the nodes of the last level in the queue.

^^^

## Rotate Image
[link]https://leetcode.com/problems/rotate-image[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                # Swap elements to transpose the matrix
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

            # Reverse the row as part of the rotation
            matrix[i] = matrix[i][::-1]
</code></pre>

* Time complexity: O(n^2), where n is the number of rows (or columns) in the matrix, or O(m), where m is the number of elements.
* Space complexity: O(1), as the rotation is done in place.

^^^

## Image Smoother
[link]https://leetcode.com/problems/image-smoother[/link]
[tag]Easy[/tag]

Fun fact: I lost count of how many times I got variants of this question in interviews. It's not a typical interview question for generic software engineering roles, but it is very common for roles related to image processing and computer vision. Similar popular questions include implementing convolution with custom kernels and various types of padding.

<pre><code class="language-python">class Solution:
    def imageSmoother(self, img):
        # Get the dimensions of the image
        m, n = len(img), len(img[0])

        # Initialize a results matrix with the same dimensions as the image
        res = [[0] * n for _ in range(m)]

        # Set the kernel size for smoothing (3x3 kernel with half kernel size 1)
        k = 1

        # Iterate through each pixel in the image
        for i in range(m):
            for j in range(n):
                total, count = 0, 0  # Initialize total and count for averaging

                # Iterate through the kernel
                for dx in range(-k, k + 1):  # Move in row
                    for dy in range(-k, k + 1):  # Move in column
                        x, y = i + dx, j + dy  # Coordinates of the neighboring pixel

                        # Check if the neighboring pixel is within the bounds of the image
                        if 0 <= x < m and 0 <= y < n:
                            total += img[x][y]  # Add the pixel's value to total
                            count += 1  # Increment count

                # Assign the average value to the current pixel in the result matrix
                res[i][j] = total // count

        return res
</code></pre>

* Time complexity: O(m * n * k^2), where m is the number of rows, n is the number of columns, and k is the kernel size. Since k is constant in this example, the complexity simplifies to O(m * n).
* Space complexity: O(m * n), since we create a result matrix of the same size as the input matrix.

^^^

## Range Sum Query 2D - Immutable
[link]https://leetcode.com/problems/range-sum-query-2d-immutable/description/[/link]
[tag]Medium[/tag]

Fun fact: This is also not a typical interview question for generic software engineering roles, but it is extremely common in roles related to image processing and computer vision. It is also known as an **integral image**, and is used for efficiently computing certain image filters. For example, it was used in the Viola–Jones object detection framework, before convolutional neural nets and transformers took over the field of computer vision. It is still used in image processing to achieve certain image blurring effects. It's an efficient way to implement a variable-sized box filter. It can also be used to approximate a Gaussian filter, by repeatedly applying a box filter.

<pre><code class="language-python">class NumMatrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.sums = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        self.computeSummedAreaTable()

    # Compute the summed area table (also known as Integral Image)
    def computeSummedAreaTable(self):
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[0])):
                self.sums[r][c] = self.matrix[r][c]  # copy the original matrix value
                # Add top and left values, subtract the top-left overlap to avoid double-counting
                if r > 0 and c > 0:
                    self.sums[r][c] += self.sums[r-1][c] + self.sums[r][c-1] - self.sums[r-1][c-1]
                elif r > 0:  # For the first column, just add the value from above
                    self.sums[r][c] += self.sums[r-1][c]
                elif c > 0:  # For the first row, just add the value from the left
                    self.sums[r][c] += self.sums[r][c-1]

    # Compute the sum of the rectangular region
    def sumRegion(self, row1, col1, row2, col2):
        result = self.sums[row2][col2]  # Start with the bottom-right value of the region
        # Adjust the sum based on the boundaries of the region
        if row1 > 0 and col1 > 0:
            result -= self.sums[row2][col1-1] + self.sums[row1-1][col2] - self.sums[row1-1][col1-1]
        elif row1 > 0:  # Adjust for the top boundary
            result -= self.sums[row1-1][col2]
        elif col1 > 0:  # Adjust for the left boundary
            result -= self.sums[row2][col1-1]

        return result
</code></pre>

* Time complexity: O(M*N), for initialization where M is the number of rows and N is the number of columns. O(1) for _sumRegion_.
* Space complexity: O(M*N), which is the size of the summed area table.

^^^

## Flood Fill
[link]https://leetcode.com/problems/flood-fill[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
    def floodFill(self, image, sr, sc, color):
        def fill(sr, sc, cur):
            # Check if the current pixel is out of bounds
            if sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
                return

            # Check if the current pixel is not of the target color or already filled
            if cur != image[sr][sc] or image[sr][sc] == color:
                return

            # Fill the current pixel
            image[sr][sc] = color

            # Recursively fill the adjacent pixels
            fill(sr-1, sc, cur)
            fill(sr+1, sc, cur)
            fill(sr, sc-1, cur)
            fill(sr, sc+1, cur)

        # Initiate filling
        fill(sr, sc, image[sr][sc])
        return image
</code></pre>

* Time complexity: O(m * n), where m and n are rows and cols. We visit every pixel at most once.
* Space complexity: O(depth of the recursion stack), which is always less than O(m * n).

^^^

## Number of Islands
[link]https://leetcode.com/problems/number-of-islands[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def numIslands(self, grid):
        # This is very similar to the fill function in the previous question.
        def fill(r, c):
            # Check if current position is out of grid bounds
            if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                return

            # Check if current position is not land ('1')
            if grid[r][c] != "1":
                return

            # Mark current land cell as visited by changing '1' to '2'
            grid[r][c] = "2"

            # Recursively visit all adjacent cells (up, down, left, right)
            fill(r + 1, c)
            fill(r - 1, c)
            fill(r, c + 1)
            fill(r, c - 1)

        # Initialize counter for number of islands
        counter = 0

        # Iterate over each cell in the grid
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                # If cell is a part of an island ('1'), increment counter
                if grid[r][c] == "1":
                    counter += 1
                    # Use recursive fill to mark the entire island as visited
                    fill(r, c)

        # Return the total count of islands found
        return counter
</code></pre>

* Time complexity: O(m * n), where m and n are rows and cols. We visit every cell in the grid at most once.
* Space complexity: O(depth of the recursion stack), which is always less than O(m * n).

^^^

## Rotting Oranges
[link]https://leetcode.com/problems/rotting-oranges[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def orangesRotting(self, grid):
        # Return -1 if grid is empty
        if not grid:
            return -1

        # Initialize rows and columns
        rows, cols = len(grid), len(grid[0])
        fresh_count = 0
        rotten = deque()

        # Count fresh oranges and add rotten oranges to the queue
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    fresh_count += 1

        # BFS loop: while there are rotten oranges and fresh oranges
        minutes = 0
        while rotten and fresh_count > 0:
            # Increment minutes for each level (minute) processed
            minutes += 1
            for _ in range(len(rotten)):
                # Get the current rotten orange's coordinates
                curr_x, curr_y = rotten.popleft()
                # Check all 4 directions around the rotten orange
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x, y = curr_x + dx, curr_y + dy
                    # Make adjacent fresh oranges rotten
                    if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 1:
                        fresh_count -= 1
                        grid[x][y] = 2
                        # Add the new rotten orange's coordinates to the queue
                        rotten.append((x, y))

        # Return the number of minutes if no fresh oranges are left, otherwise -1
        return minutes if fresh_count == 0 else -1
</code></pre>

* Time complexity: O(m * n), where m is the number of rows and n is the number of columns. In the worst case, every cell in the grid will be visited.
* Space complexity: O(m * n). Size of the queue in the worst case, where all cells are filled with rotten oranges initially.

^^^

## Best Time to Buy and Sell Stock
[link]https://leetcode.com/problems/best-time-to-buy-and-sell-stock[/link]
[tag]Easy[/tag]

<pre><code class="language-python">class Solution:
""" Iterate through the list once, tracking the minimum price
    so far and the maximum profit that can be obtained. """

    def maxProfit(self, prices):
        min_price = prices[0]
        max_profit = 0

        for price in prices:
            min_price = min(min_price, price)
            profit = price - min_price
            max_profit = max(max_profit, profit)

        return max_profit
</code></pre>

Time complexity: O(n). We iterate through all the prices once.
Space complexity: O(1), since we are using a fixed amount of extra space for the variables.

^^^

## Middle of the Linked List
[link]https://leetcode.com/problems/middle-of-the-linked-list[/link]
[tag]Easy[/tag]

<pre><code class="language-python">""" The slow pointer moves by one node and the fast pointer by two.
When the fast pointer reaches the end, the slow pointer is at the middle node. """
class Solution:
    def middleNode(self, head):
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow
</code></pre>

* Time complexity: O(N), where N is the number of nodes. We are traversing the list only once.
* Space complexity: O(1), as we are using a constant amount of extra space (the two pointers).

^^^

## Reorder List
[link]https://leetcode.com/problems/reorder-list[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def reorderList(self, head):
        if not head:
            return None

        # Find the middle node
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        cur = slow

        # Reverse the second half
        prev = None
        while cur:
            cur.next, prev, cur = prev, cur, cur.next

        list_second = prev

        # Interleave the two halves
        list_first = head
        while list_second.next:
            list_first.next, list_second.next, list_second, list_first = (
                list_second,
                list_first.next,
                list_second.next,
                list_first.next,
            )
</code></pre>

* Time complexity: O(n), where n is the number of nodes in the list. Each part of the process (finding the middle, reversing the second half, and merging) is linear in time.
* Space complexity: O(1), as the reordering is done in place with a fixed number of extra variables.

^^^

## Maximum Subarray
[link]https://leetcode.com/problems/maximum-subarray[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def maxSubArray(self, nums):
        # max_sum stores the maximum sum found so far.
        # cur_sum stores the current sum of the subarray.
        max_sum = cur_sum = nums[0]

        # Iterate over the rest of the array.
        for n in nums[1:]:
            # Decide whether to start a new subarray at the current
            # element or to continue with the existing subarray.
            cur_sum = max(n, cur_sum + n)

            # Update max_sum if the current sum is greater.
            max_sum = max(max_sum, cur_sum)

        # Return the maximum sum of a contiguous subarray found.
        return max_sum
</code></pre>

* Time complexity: O(n), where n is the length of the input array.
* Space complexity: O(1), as the algorithm uses a constant amount of extra space for the variables.

^^^

## Maximum Product Subarray
[link]https://leetcode.com/problems/maximum-product-subarray[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def maxProduct(self, nums):
        # max_prod is the maximum product found so far.
        # cur_min and cur_max are the min and max products ending at the current position.
        max_prod = cur_min = cur_max = nums[0]

        for n in nums[1:]:
            # The new cur_max is the maximum of three values:
            # 1. The current number n (starting a new subarray here).
            # 2. Product of n and previous cur_max.
            # 3. Product of n and previous cur_min (useful when n is negative).
            # The new cur_min is similarly calculated but for finding the minimum.
            cur_max, cur_min = max(n, cur_max * n, cur_min * n), min(n, cur_max * n, cur_min * n)

            # Update max_prod if the current cur_max is greater than the current max_prod.
            max_prod = max(max_prod, cur_max)

        # Return the maximum product subarray found.
        return max_prod
</code></pre>

* Time complexity: O(n), where n is the length of the input array.
* Space complexity: O(1). We use a fixed number of variables (max_prod, cur_min, cur_max) regardless of the input size.

^^^

## Longest Increasing Subsequence
[link]https://leetcode.com/problems/longest-increasing-subsequence[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def lengthOfLIS(self, nums):

        # Based on Patience Sorting Algorithm.

        tails = []
        for num in nums:
            # Find the index in tails where num should be inserted
            # bisect_left returns the leftmost place in the sorted list to insert num
            idx = bisect.bisect_left(tails, num)

            # If idx is equal to the length of tails, it means num is greater than all elements in tails
            # Therefore, num extends the longest subsequence formed so far
            if idx == len(tails):
                tails.append(num)
            # If num is not greater, replace the element at idx with num
            # This step ensures that the element at idx is the smallest possible
            else:
                tails[idx] = num

        return len(tails)

</code></pre>

* Time complexity: O(N log N), where N is the length of the input list nums. For each of element in nums, we do a binary search with O(log N) complexity.
* Space complexity: O(N) due to the additional list tails .

^^^

## Longest Consecutive Sequence
[link]https://leetcode.com/problems/longest-consecutive-sequence[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def longestConsecutive(self, nums):

        nums = set(nums)
        longest = 0

        for num in nums:
            # Check if this is the beginning of a new consecutive sequence
            # i.e. there is no num - 1
            if num - 1 not in nums:
                cur_num = num

                # Expand the sequence forward, look for num + 1 iteratively
                while cur_num + 1 in nums:
                    cur_num += 1

                # cur_num is now the end of the consecutive sequence
                longest = max(longest, cur_num - num + 1)

        return longest
</code></pre>

* Time complexity: O(n). Even though there are nested loops, the inner while loop runs only once for each number in the sequence. We start counting a new streak only if the current number is the start of a sequence.
* Space complexity: O(n). We use a set to store all the elements of nums.

^^^

## Daily Temperatures
[link]https://leetcode.com/problems/daily-temperatures[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def dailyTemperatures(self, temperatures):
        # Array of days to wait for warmer temperatures after each day i.
        # filled with 0s (default values indicating no warmer day found)
        res = [0] * len(temperatures)

        stack = []
        for i, t in enumerate(temperatures):
            # Calculate the number of days until a warmer temperature
            # and update the result at the index 'last'
            while stack and t > temperatures[stack[-1]]:
                last = stack.pop()
                res[last] = i - last

            # Push the current index onto the stack. This index represents
            # a temperature for which we are yet to find a warmer day.
            stack.append(i)

        return res
</code></pre>

* Time complexity: O(N). Each element is processed once when it is added to the stack and at most once when it is popped from the stack.
* Space complexity: O(N). In worst case, all temperatures could be in non-decreasing order, causing all indices to be stored in the stack until the very end.

^^^

## Sliding Window Maximum
[link]https://leetcode.com/problems/sliding-window-maximum[/link]
[tag]Hard[/tag]

<pre><code class="language-python">from collections import deque

class Solution:
    def maxSlidingWindow(self, nums, k):

        # Use a deque to store indices of elements in the array.
        # The end of the deque always stores the index of the maximum element in the current window.

        dq = deque()
        result = []

        for i in range(len(nums)):
            # Remove indices from the back of the deque if they point to elements
            # smaller than the current element, as they are no longer useful
            while dq and nums[i] > nums[dq[-1]]:
                dq.pop()

            # Add the index of the current element to the deque
            dq.append(i)

            # Remove the front element of the deque if it's outside the current window
            if dq[0] < i - k + 1:
                dq.popleft()

            # If the current window size is at least k, add the maximum element
            # (which is at the front of the deque) to the result list
            if i >= k - 1:
                result.append(nums[dq[0]])

        return result
</code></pre>

* Time complexity: O(n), where n is the number of elements in nums, since each element is added and removed from the deque at most once.
* The space complexity: O(k), due to the deque storing at most k elements (the size of the window).

^^^

## Minimum Window Substring
[link]https://leetcode.com/problems/minimum-window-substring[/link]
[tag]Hard[/tag]

<pre><code class="language-python">from collections import Counter

class Solution:
    def minWindow(self, s, t):
        # We are going to use an expanding and shrinking sliding window to find the 
        # shortest substring of s such that every character in t is included in the window. 

        # Counters for characters in t and the current window
        t_counter = Counter(t)
        window_counter = Counter()

        # Initialize left and right pointers for the sliding window
        left = right = 0

        # Variables to track the size and content of the minimum window
        min_window_size = float('inf')
        min_window = ""

        # Variable to count the number of character types matching between window and t
        match_count = 0

        while right < len(s):
            # Include current character in the window
            window_counter[s[right]] += 1

            # If the current character's count matches in t, increment match_count
            if window_counter[s[right]] == t_counter[s[right]]:
                match_count += 1

            # Try to minimize the window while it still contains all characters of t
            while left <= right and match_count == len(t_counter):
                # Update minimum window if a smaller valid window is found
                if right - left + 1 < min_window_size:
                    min_window_size = right - left + 1
                    min_window = s[left:right+1]

                # Remove the leftmost character from the window
                window_counter[s[left]] -= 1

                # If this causes a character type to fall below its required count in t, decrement match_count
                if window_counter[s[left]] < t_counter[s[left]]:
                    match_count -= 1

                # Move the left pointer to the right, shrinking the window
                left += 1

            # Move the right pointer to the right, expanding the window
            right += 1

        return min_window
</code></pre>

* Time complexity: O(M + N), where M is the length of string s and N is the length of string t.
* Space complexity: O(M + N) due to the extra space used for the two counters (hash tables).

^^^

## Longest Substring Without Repeating Characters
[link]https://leetcode.com/problems/longest-substring-without-repeating-characters[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def lengthOfLongestSubstring(self, s):

        # Dictionary to store the last positions of characters
        seen = {}
        start_idx = 0 # start index of the current substring
        max_sslen = 0 # length of the longest substring

        for i, c in enumerate(s):
            # If we have seen this character before then we are at the
            # end of a considered substring without repeating characters.
            # Update the maximum length and the start index, and continue.
            if c in seen:
                max_sslen = max(max_sslen, i - start_idx)
                start_idx = max(start_idx, seen[c] + 1)
            seen[c] = i

        # Return the larger of the longest substring so far and the last considered substring.
        return max(max_sslen, len(s) - start_idx)
</code></pre>

* Time complexity: O(n), where n is the length of the string.
* Space complexity: O(1), assuming a fixed-sized character set. The seen dictionary can have at most as many elements as the size of the character set.

^^^

## Longest Repeating Character Replacement
[link]https://leetcode.com/problems/longest-repeating-character-replacement[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import Counter

class Solution:
    def characterReplacement(self, s, k):
        # Counter to keep track of the frequency of each character in the current window
        char_count = Counter()
        max_freq, left = 0, 0

        # Iterate over the string with the right pointer of the sliding window
        for right in range(len(s)):
            # Update the maximum frequency of a single character in the current window
            char = s[right]
            char_count[char] += 1
            max_freq = max(max_freq, char_count[char])

            # Check if the current window is valid (i.e., can be turned into a single character string
            # by replacing at most k characters). If not, shrink the window from the left.
            if (right - left + 1 - max_freq) > k:
                char_count[s[left]] -= 1
                left += 1

        return right - left + 1
</code></pre>

* Time complexity: O(n), where n is the length of the string s. Each character in s is visited at most twice, once by the right pointer and once by the left pointer.
* Space complexity: O(1), as the char_count dictionary is bound by the size of the character set (26 letters in this example).

^^^

## Unique Paths
[link]https://leetcode.com/problems/unique-paths[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    """ Solution 1: Recursion with memoization. """
    def uniquePaths(self, m, n):
        memo = {}

        def _unique_paths(m, n):
            if (m, n) in memo:
                return memo[(m, n)]

            # Base case: If either index is 1, there's only one path
            if m == 1 or n == 1:
                return 1

            # Recursive case: The number of paths to (m, n) is the sum of
            # the paths to (m-1, n) and (m, n-1)
            memo[(m, n)] = _unique_paths(m - 1, n) + _unique_paths(m, n - 1)

            return memo[(m, n)]

        return _unique_paths(m, n)
</code></pre>

* Time Complexity: O(mn). Each cell (m, n) is computed once and then stored.
* Space Complexity: O(mn). The memoization dictionary stores a result for each of the m*n cells.

---

<pre><code class="language-python">import math
class Solution:
    """ Solution 2: Use combinatorics. """
    def uniquePaths(self, m, n):
        
        # C(n, r) = n! / (r! * (n - r)!)
        def comb(n, r):
            return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

        # Choose (m-1) downs from (m+n-2) moves
        return comb(m + n - 2, m - 1)
</code></pre>

* Time complexity: O(m + n).
* Space complexity: O(1).

^^^

## Jump Game
[link]https://leetcode.com/problems/jump-game[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def canJump(self, nums):
        # This keeps track of the minimum index from which the last index can be reached.
        min_idx = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= min_idx:
                # This means we can reach the end from index i
                min_idx = i

        return min_idx == 0
</code></pre>

* Time complexity: O(N).
* Space complexity is O(1).

^^^

## Search a 2D Matrix
[link]https://leetcode.com/problems/search-a-2d-matrix[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def searchMatrix(self, matrix, target):

        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1

        # Start binary search
        while left <= right:
            # Convert the middle index to 2D indices in the matrix
            # Treat the matrix as a flattened array
            mid = (left + right) // 2
            mid_value = matrix[mid // n][mid % n]

            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
</code></pre>

* Time complexity: O(log(mn)). We do a binary search over a virtual array of length mn.
* Space complexity: O(1).

^^^

## Set Matrix Zeroes
[link]https://leetcode.com/problems/set-matrix-zeroes[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def setZeroes(self, matrix):
        # Two sets to keep track of rows and columns that need to be zeroed
        rows, cols = set(), set()

        # Find zero elements and record their row and column numbers
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        # Set elements to zero if their row or column index is in rows or cols set
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in rows or j in cols:
                    matrix[i][j] = 0
</code></pre>

* Time Complexity: O(m * n), where m and n are the input matrix dimensions.
* Space Complexity: O(m + n), in the worst case, we might need to store all rows and all columns in the rows and cols sets. 
* Follow-up: To achieve an O(1) space solution, we can use the first row and the first column of the matrix as markers to indicate which rows and columns need to be set to zero. This approach allows us to avoid using extra space for sets.

^^^

## Letter Combinations of a Phone Number
[link]https://leetcode.com/problems/letter-combinations-of-a-phone-number[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def letterCombinations(self, digits):

        if not digits:
            return []
        
        # Mapping of digits to letters
        mapping = {
            "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
            "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
        }

        result = []
        
        # Helper function to perform the recursive backtracking
        def backtrack(combination, next_digits):
            # If there are no more digits to check, add the current combination to the result
            if len(next_digits) == 0:
                result.append(combination)
            else:
                # For the current digit, iterate over each letter it can represent
                # Add the current letter to the combination and proceed with the remaining digits
                for letter in mapping[next_digits[0]]:
                    backtrack(combination + letter, next_digits[1:])
        
        backtrack("", digits)
        return result
</code></pre>


* Time complexity: O(4^N * N). We are building a string of length N for each combination. There are at most 4^N combinations, since each digit maps to at most 4 letters.
* Space complexity: O(N) due to the recursion call stack. The depth of the recursion tree can go up to N.

^^^

## Decode Ways
[link]https://leetcode.com/problems/decode-ways[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def numDecodings(self, s):
        n = len(s)
        
        # If the string is empty or starts with '0', no decoding is possible
        if n == 0 or s[0] == '0':
            return 0

        # Dynamic programming table where dp[i] represents the number of ways
        # to decode the string up to character i
        dp = [0] * (n + 1)
        
        # Base cases
        dp[0] = 1  # Empty string has one way to decode
        dp[1] = 1  # String of length 1 has one way to decode unless it starts with '0'

        for i in range(2, n + 1):
            # Single digit decoding (valid if not '0')
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]

            # Two digit decoding (valid if between '10' and '26')
            two_digit = int(s[i - 2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]

        return dp[n]
</code></pre>

* Time complexity: O(n), where n is the length of the string.
* Space complexity: O(n) because of the dp array.

^^^

## Word Break
[link]https://leetcode.com/problems/word-break[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def wordBreak(self, s, wordDict):
        # Convert wordDict to a set for O(1) lookups
        wordSet = set(wordDict)

        # The length is len(s) + 1 because we need an extra slot for the base case.
        # Base case: an empty string can always be segmented
        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(1, len(s) + 1):
            # Check each possible starting index of the substring.
            for j in range(i):
                # Check if the substring j to i is in wordSet,
                # and if the substring up to start index j can be segmented
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    # Once we find a valid segmentation, no need to continue checking
                    break

        return dp[-1]
</code></pre>

* Time complexity of this solution is O(n^2 * m), where n and m are the lengths of the input string and the longest word in wordDict.
* Space complexity: O(n), for the dp array we are using.

^^^

## Word Search
[link]https://leetcode.com/problems/word-search[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def exist(self, board, word):

        def dfs(i, j, k):
            """
            Depth-first search (DFS) to find the word in the board.
            k: Current index in the word
            """
            # Check for out of bounds or character mismatch
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
                return False

            # Check if all characters in the word are found
            if k == len(word) - 1:
                return True

            # Temporarily mark the current board cell to avoid revisiting
            tmp = board[i][j]
            board[i][j] = "/"

            # Explore all four directions (up, down, left, right)
            res = (dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or 
                   dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1))

            # Restore the original character in the board cell
            board[i][j] = tmp

            return res

        # Iterate over each cell in the board as a potential starting point
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True

        return False
</code></pre>

* Time complexity: O(N * M * 4^L). For each cell in the board (N * M), the algorithm performs a depth-first search (DFS) that could, in the worst case, explore 4 directions at each step (4^L, where L is the length of the word).
* Space Complexity: O(L). The primary space consumption is the call stack for the DFS, which goes as deep as the length of the word (L).

^^^

## Course Schedule
[link]https://leetcode.com/problems/course-schedule[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def canFinish(self, numCourses, prerequisites):
        # Create a graph from prerequisites, a list of edges for each node
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[course].append(prereq)

        # visited[i] = 0 means not visited, 1 means visited, -1 means currently visiting
        visited = [0] * numCourses

        # DFS function to detect cycle
        def has_cycle(course):
            if visited[course] == -1:
                # Found a cycle
                return False
            if visited[course] == 1:
                # Already visited, no cycle here
                return True

            # Mark as currently visiting
            visited[course] = -1

            # Visit all prerequisites
            for prereq in graph[course]:
                if not has_cycle(prereq):
                    return False

            # Mark as visited
            visited[course] = 1
            return True

        # Check each course for a cycle
        for course in range(numCourses):
            if not has_cycle(course):
                return False

        return True
</code></pre>

* Time complexity: O(N + P), where N is the number of courses and P is the number of prerequisites. We iterate through each course and its prerequisites.
* Space complexity: O(N + P), for the graph and the visited list.

^^^

## Word Ladder
[link]https://leetcode.com/problems/word-ladder[/link]
[tag]Hard[/tag]

<pre><code class="language-python">from collections import deque
import string

class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        # Convert the wordList into a set for O(1) look-ups.
        wordSet = set(wordList)

        # If the endWord is not in the wordSet, return 0.
        if endWord not in wordSet:
            return 0

        # Initialize a queue for BFS and add the beginWord.
        # The tuple contains the word and its level (distance from beginWord).
        queue = deque([(beginWord, 1)])

        while queue:
            word, length = queue.popleft()

            # If this word is the endWord, return the length.
            if word == endWord:
                return length

            # Iterate through each character in the word and
            # change the ith character to every possible lowercase alphabet.
            for i in range(len(word)):
                for char in string.ascii_lowercase:
                    next_word = word[:i] + char + word[i+1:]

                    # If the new word is in the wordSet
                    if next_word in wordSet:
                        # Add this word to the queue with an increased length
                        queue.append((next_word, length + 1))
                        # Remove the word from the set to prevent revisiting
                        wordSet.remove(next_word)

        return 0
</code></pre>

* Time complexity: O(N*M^2). For each of the N words in the wordList, we iterate through its M characters. For each character in a word, we try replacing it with every possible lowercase letter. The number of lowercase letters is a constant factor. However, the act of creating a new word (word[:i] + char + word[i + 1:]) has a complexity of O(M) because it involves slicing and concatenation of strings, leading to an overall time complexity of O(N M^2).
* Space complexity: O(MN): In the worst case all words are in the queue.

^^^

## LRU Cache
[link]https://leetcode.com/problems/lru-cache[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        else:
            # Manually move the key to the end
            # LeetCode doesn't support self.cache.move_to_end(key)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key, value):
        if key in self.cache:
            # Remove the existing key before re-inserting it
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Pop the first item (least recently used)
            self.cache.popitem(last=False)
        self.cache[key] = value
</code></pre>

* Time complexity: O(1). Dictionary access and moving an element to the end of an OrderedDict are O(1) operations.
* Space complexity: O(capacity).

This solution might seem like cheating because the OrderedDict already contains everything needed to create an LRU cache. Typically, this problem would be solved using a regular dictionary coupled with a doubly linked list to track the usage order. This approach is likely similar to the underlying implementation of OrderedDict.

^^^

## Container With Most Water
[link]https://leetcode.com/problems/container-with-most-water[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def maxArea(self, height):
        max_area = 0

        # Use two pointers, starting from the leftmost and rightmost edges
        left = 0
        right = len(height) - 1

        while left < right:
            # Calculate the area formed between the two pointers
            area = (right - left) * min(height[left], height[right])
            max_area = max(max_area, area)

            # Move the pointer which points to the shorter line towards the other,
            # since this might increase the area
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area
</code></pre>

* Time complexity: O(n). The while loop runs a maximum of n times, as each iteration moves one of the pointers (left or right) closer to the other.
* Space complexity: O(1). The algorithm uses a fixed number of variables.

^^^

## Trapping Rain Water
[link]https://leetcode.com/problems/trapping-rain-water[/link]
[tag]Hard[/tag]

<pre><code class="language-python">"""At each step, we calculate the potential water trapped at the current position,
based on the maximum height encountered so far from both ends. This is because the
amount of water that can be trapped at any point is determined by the height of the
shorter of the two sides (left and right max heights).

The pointers move towards each other, and the process continues until they meet.
This ensures that the water trapped at each bar is accounted for, by comparing 
the height of each bar with the minimum of the maximum heights encountered from
both directions up to that point. """

class Solution:
    def trap(self, height):
        left, right = 0, len(height) - 1
        water = 0
        left_max, right_max = height[left], height[right]

        while left < right:
            if height[left] < height[right]:
                left += 1
                left_max = max(left_max, height[left])
                water += left_max - height[left]                
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water += right_max - height[right]

        return water
</code></pre>

* Time complexity: O(N), where N is the number of elements in the height list.
* Space complexity: O(1).

^^^

## Insert Interval
[link]https://leetcode.com/problems/insert-interval[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def insert(self, intervals, new_interval):
        result = []

        for interval in intervals:
            # If the current interval ends before the new interval starts, add it to the result
            if interval[1] < new_interval[0]:
                result.append(interval)
            # If the current interval starts after the new interval ends,
            # add the new interval to the result and update the new interval to be the current one
            elif new_interval[1] < interval[0]:
                result.append(new_interval)
                new_interval = interval
            # If the intervals overlap, update the new interval to be the merged interval
            else:
                new_interval[0] = min(interval[0], new_interval[0])
                new_interval[1] = max(interval[1], new_interval[1])

        result.append(new_interval)
        return result
</code></pre>

* Time complexity: O(n).
* Space complexity: O(n), to store all given intervals plus the new interval in the result list.

^^^

## Merge Intervals
[link]https://leetcode.com/problems/merge-intervals/[/link]
[tag]Medium[/tag]

Fun fact: I used this algorithm to organize videos from my backyard camera. First, I ran a person detector to identify frames with people. Next, I identified the segments where people were present and discarded the parts without humans. To address gaps when people temporarily leave the frame or the detector misses them, I expanded the intervals with people. This resulted in overlapping video intervals. I used this algorithm to merge those intervals. Finally, I used ffmpeg to losslessly cut the videos into those segments.

<pre><code class="language-python">class Solution:
    def merge(self, intervals):
        # Sort intervals based on start time
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # If the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # Otherwise, there is overlap, so we merge the current and previous intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
</code></pre>

* Time complexity: O(NlogN) because of sorting, where N is the number of intervals. The merging process is O(N) since it involves a single pass through the sorted intervals.
* Space complexity: O(N) if we consider the space for the output.

^^^

## K Closest Points to Origin
[link]https://leetcode.com/problems/k-closest-points-to-origin[/link]
[tag]Medium[/tag]

<pre><code class="language-python">""" Solution 1: Sort the points based on their Euclidean distance from the origin.
Return the first k elements of the sorted list, which are the k closest points."""
class Solution:
    def kClosest(self, points, k):
        points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
        return points[:k]
</code></pre>

* Time complexity: O(n log n). The sort function in Python has a time complexity of O(n log n).
* Space complexity: O(1), as the sorting is done in-place.

---

<pre><code class="language-python">""" Solution 2: Use a heap to keep track of the k closest points to the origin.
Using negative distance, we make sure that the farthest point among the closest
ones is always at the top of the heap, so that we can pop it easily."""

import heapq

class Solution:
    def kClosest(self, points, k):
        heap = []
        for x, y in points:
            dist = -(x ** 2 + y ** 2)
            if len(heap) < k:
                heapq.heappush(heap, (dist, x, y))
            else:
                heapq.heappushpop(heap, (dist, x, y))
        return [(x, y) for (dist, x, y) in heap]
</code></pre>

* Time complexity: O(n log k), where n is the number of points. Each insert operation into the heap takes O(log k) time, and this is done for each of the n points.
* Space complexity: O(k) for storing the heap.

^^^

## Longest Palindromic Substring
[link]https://leetcode.com/problems/longest-palindromic-substring[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def longestPalindrome(self, s):
        # Main idea: iterate over each character,
        # treat them as the center of the palindrome and expand

        def expand(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1

        start, end = 0, 0

        for i in range(len(s)):
            left_odd, right_odd = expand(s, i, i)
            left_even, right_even = expand(s, i, i + 1)

            left = min(left_odd, left_even)
            right = max(right_odd, right_even)

            if right - left > end - start:
                start = left
                end = right

        return s[start:end + 1]
</code></pre>

* Time complexity: O(n^2). In worst case, for each character, we might expand in both directions until we reach the ends of the string.
* Space complexity: O(1). We only use a few variables for keeping track of the current longest palindrome.

^^^

## Generate Parentheses
[link]https://leetcode.com/problems/generate-parentheses[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def generateParenthesis(self, n):
        # This function generates all combinations of well-formed parentheses.

        def backtrack(s, open_count, close_count):
            # s: Current string of parentheses.
            # open_count: Number of open parentheses used.
            # close_count: Number of close parentheses used.

            # Base case: If the current string length equals 2 * n, it's a valid combination.
            if len(s) == 2 * n:
                return [s]

            results = []

            # If the number of open parentheses is less than n, add an open parenthesis.
            if open_count < n:
                results += backtrack(s + "(", open_count + 1, close_count)

            # If the number of close parentheses is less than the number of open ones, add a close parenthesis.
            # This ensures the parentheses are well-formed.
            if close_count < open_count:
                results += backtrack(s + ")", open_count, close_count + 1)

            return results

        # Start the backtracking process with an empty string and counts set to 0.
        return backtrack("", 0, 0)
</code></pre>

* Time complexity: O(4^n / sqrt(n)). The maximum depth of recursion is 2 * n, since each level of recursion adds one parenthesis, and we stop when we have 2 * n parentheses. At each step, we have two choices (adding an open parenthesis or a close parenthesis), but these choices are not always available. The actual branching factor varies, but it's less than or equal to 2. Therefore, the upper bound is O(2^(2n)) = O(4^n). However, this is a very loose upper bound. In practice, the time complexity is significantly lower due to the constraints on when we can add open or close parentheses. This is often represented as O(4^n / sqrt(n)) based on the nth Catalan number.
* Space complexity: O(n). In the worst case, the maximum depth of the recursive call stack would be 2n.

^^^

## Pow(x, n)
[link]https://leetcode.com/problems/powx-n[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def myPow(self, x, n):
        # Fast exponentiation method

        # If n is negative, convert the problem to x^-n = 1/x^n
        if n < 0:
            x = 1 / x
            n = -n

        # Helper function to recursively compute the power.
        def helper(x, n):
            # Base case: any number to the power of 0 is 1.
            if n == 0:
                return 1

            # Recursive case: divide the problem into smaller sub-problems.
            # Compute x^(n//2) once and reuse it.
            half = helper(x, n // 2)

            # If n is even, the result is half * half.
            # If n is odd, multiply an additional x to account for the odd power.
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x

        return helper(x, n)
</code></pre>

* Time complexity: O(log n), because it recursively divides the problem in half each time.
* Space complexity: O(log n), due to the space used by the recursion stack and there are log n calls in total because the problem size is halved with each call.

^^^

## Subsets
[link]https://leetcode.com/problems/subsets[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def subsets(self, nums):

        if len(nums) == 0:
            return [[]]

        # Exclude current num
        subsets_exclude = self.subsets(nums[1:])

        # Include current num
        subsets_include = []

        for subset in subsets_exclude:
            subsets_include.append(subset + [nums[0]])

        return subsets_exclude + subsets_include
</code></pre>

* Time complexity: O(2^n), where n is the number of elements in the list. Each element in the list can either be in a subset or not, leading to 2^n possible subsets.
* Space complexity: O(2^n), mainly due to the space required to store all the subsets. The space used by the recursion stack is O(n), as the maximum depth of recursion tree is n. However, the number of subsets dominates this.

^^^

## Combination Sum
[link]https://leetcode.com/problems/combination-sum[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def combinationSum(self, candidates, target):

        # Base Cases
        if target == 0:
            return [[]]
        if target < 0 or len(candidates) == 0:
            return []
        
        # Exclude current candidate
        combinations_exclude = self.combinationSum(candidates[1:], target)
        
        # Include current candidate
        combinations_include = self.combinationSum(candidates, target - candidates[0])
        
        # Add current candidate to all combinations where it is included
        for combination in combinations_include:
            combination.append(candidates[0])
        
        return combinations_exclude + combinations_include
</code></pre>

* Time complexity: O(N^(target / min(candidates))). It's exponential because of branching. The maximum depth of the recursive tree is limited by the target value. In the worst case, this could be when the smallest number in candidates is repeatedly used to reach the target. So, the depth could be approximately target / min(candidates).
* Space complexity is O(T / min(candidates) ), which is the maximum depth of the recursion tree.

^^^

## Permutations
[link]https://leetcode.com/problems/permutations[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def permute(self, nums):

        def generate_perm(nums, perm=[]):
            # Base case: if nums is empty, return the permutation
            if not nums: 
                return [perm]

            results = []

            # Iterate through each number in nums
            for i in range(len(nums)): 
                # Recursively call generate_perm with the remaining numbers
                # and the current number added to the permutation
                results += generate_perm(nums[:i] + nums[i+1:], perm + [nums[i]]) 
            return results

        # Call the helper function with the initial list of numbers
        return generate_perm(nums)
</code></pre>

* Time complexity: O(N * N!). There are N choices for the first element, N-1 for the second, and so on, leading to N! total permutations. For each permutation, the algorithm constructs a new list by concatenating lists (nums[:i] + nums[i+1:]) and appending elements (perm + [nums[i]]). The list concatenation and append operations take O(N) time.
* Space complexity: O(N!), due to the storage requirements of all the generated permutations. The space complexity includes the depth of the recursive call stack, which in the worst case can be O(N), where N is the length of the input list.

^^^

## Evaluate Reverse Polish Notation
[link]https://leetcode.com/problems/evaluate-reverse-polish-notation[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def evalRPN(self, tokens):
        # Stack to store numbers
        stack = []

        # Iterate through each token
        for token in tokens:
            if token in "+-*/":
                # Pop two numbers for the operation
                num2 = stack.pop()
                num1 = stack.pop()

                # Apply the operation and push the result back
                if token == '+':
                    stack.append(num1 + num2)
                elif token == '-':
                    stack.append(num1 - num2)
                elif token == '*':
                    stack.append(num1 * num2)
                elif token == '/':
                    stack.append(int(float(num1) / num2))
            else:
                # Push number onto stack
                stack.append(int(token))

        # The result is the only element left in the stack
        return stack[0]
</code></pre>

* Time complexity: O(n), where n is the number of tokens. We iterate through each token exactly once.
* Space complexity: O(n), in the worst case, the stack might contain all the tokens if they are all numbers. However, in practice, the stack size will be considerably less than n for most inputs since operators reduce the stack size.

^^^

## Lowest Common Ancestor of a Binary Tree
[link]https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def lowestCommonAncestor(self, root, p, q):
        # Base case: if root is None or root is one of p or q
        if root is None or root == p or root == q:
            return root

        # Search in the left subtree
        left = self.lowestCommonAncestor(root.left, p, q)
        # Search in the right subtree
        right = self.lowestCommonAncestor(root.right, p, q)

        # If both left and right are not None, root is the LCA
        if left and right:
            return root
        # If only one of left or right is not None, return the non-None one
        return left if left else right
</code></pre>

* Time Complexity: O(N), where N is the number of nodes in the binary tree. In the worst case, the function might have to visit all nodes of the tree (especially if the tree is skewed).
* Space Complexity: O(H), where H is the height of the binary tree. This is due to the recursive nature of the algorithm, which consumes stack space. In the worst case (a skewed tree), the space complexity can become O(N), but in a balanced tree, it would be O(log N).

^^^

## Binary Tree Right Side View
[link]https://leetcode.com/problems/binary-tree-right-side-view[/link]
[tag]Medium[/tag]

<pre><code class="language-python">from collections import deque

class Solution:
    def rightSideView(self, root):
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_length = len(queue)
            for i in range(level_length):
                node = queue.popleft()
                # Add the rightmost element of the current level to the result
                if i == level_length - 1:
                    result.append(node.val)
                # Add left and right children of the current node to the queue
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result
</code></pre>

* Time complexity: O(N). Each node is visited once. Queue operations are constant time.
* Space complexity: O(D), where D is the diameter of the tree. In the worst case, the queue will hold all nodes at the level with the maximum number of nodes. The diameter provides a tight upper bound for this, which is always lower than O(N).

^^^

## Binary Tree Maximum Path Sum
[link]https://leetcode.com/problems/binary-tree-maximum-path-sum[/link]
[tag]Hard[/tag]

<pre><code class="language-python">class Solution:
    def maxPathSum(self, root):
        def max_gain(node):
            # Helper function for depth-first-search
            nonlocal max_sum
            if not node:
                return 0

            # Compute the maximum path sum starting from the left and right child
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # Update the global maximum path sum
            max_sum = max(max_sum, node.val + left_gain + right_gain)

            # Return the maximum gain the current node can contribute to its parent
            return node.val + max(left_gain, right_gain)

        max_sum = float('-inf')
        max_gain(root)
        return max_sum
</code></pre>

* Time complexity: O(N). We visit each node exactly once.
* Space complexity: O(H), where H is the height of the tree. This space is used by the call stack during the recursive calls. In the worst case (a skewed tree), the height of the tree can be O(N), leading to O(N) space complexity. In a balanced tree, it would be O(log N).

^^^

## Implement Trie (Prefix Tree)
[link]https://leetcode.com/problems/implement-trie-prefix-tree[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class TrieNode:
    def __init__(self):
        self.children = {}  # Dictionary to hold child nodes
        self.isEndOfWord = False  # Flag to mark the end of a word

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            # For each character in the word, insert it into the trie
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isEndOfWord = True  # Mark the end of the word

    def search(self, word, is_exact=True):
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for char in word:
            # For each character in the word, search in the trie
            if char not in node.children:
                return False
            node = node.children[char]

        # Return True if it's the end of a word and it needs to be exact
        return node.isEndOfWord if is_exact else True  

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        return self.search(prefix, is_exact=False)
</code></pre>

* Time complexity: For insert: O(n), where n is the length of the word to insert. For search and startsWith: O(m), where m is the length of the word or prefix to search.
* Space complexity: O(T * L), where T is the total number of trie nodes and L is the average length of the words. In the worst case, this can be quite large if there are many words with very little common prefix. However, in practice, the common prefixes shared among words help reduce space usage.

^^^

## Clone Graph
[link]https://leetcode.com/problems/clone-graph[/link]
[tag]Medium[/tag]

<pre><code class="language-python">"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node):
        # This helper function is used to clone each node 
        # and to keep track of already cloned nodes.
        def clone_node(node, visited={}):
            if not node:
                return None

            # Create a clone of the current node and add it to the visited dictionary.
            node_clone = Node(node.val)
            visited[node.val] = node_clone

            # Iterate over the neighbors of the current node.
            for n in node.neighbors:
                # If the neighbor is already cloned, simply append the cloned neighbor.
                if n.val in visited:
                    node_clone.neighbors.append(visited[n.val])
                else:
                    # If the neighbor is not visited, recursively clone it and append the clone.
                    node_clone.neighbors.append(clone_node(n, visited))

            return node_clone

        # Start cloning from the given node.
        return clone_node(node)
</code></pre>

* Time complexity: O(N) because each node is visited exactly once.
* Space complexity: O(N) due to the storage required for cloned_nodes and the recursive stack space in the worst case.

^^^

## Edit Distance
[link]https://leetcode.com/problems/edit-distance[/link]
[tag]Medium[/tag]

We will create a 2D array dp where dp[i][j] represents the minimum number of operations required to convert the first i characters of word1 to the first j characters of word2. We will fill this array in a bottom-up manner. The operations include insert, delete, and replace.

<pre><code class="language-python">class Solution:
    def minDistance(self, word1, word2):
        # Getting the lengths of both words
        m, n = len(word1), len(word2)

        # Initialize a 2D array with dimensions (m+1) x (n+1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases: 
        # If word2 is empty, we need to delete all characters from word1
        for i in range(m + 1):
            dp[i][0] = i
        # If word1 is empty, we need to insert all characters of word2
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the dp array
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # If the characters are the same, no operation is needed
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Consider all three operations and choose the minimum
                    dp[i][j] = 1 + min(dp[i - 1][j],    # Delete
                                       dp[i][j - 1],    # Insert
                                       dp[i - 1][j - 1]) # Replace

        return dp[m][n]
</code></pre>

* Time complexity: O(m * n), because we need to fill up a 2D array of size m * n.
* Space complexity: O(m * n) for the 2D array dp.

^^^

## Find Median from Data Stream
[link]https://leetcode.com/problems/find-median-from-data-stream[/link]
[tag]Hard[/tag]

<pre><code class="language-python">import heapq

class MedianFinder:

    def __init__(self):
        # Initialize two heaps: max_heap for lower half and min_heap for upper half
        self.max_heap, self.min_heap = [], []

    def addNum(self, num):
        # Add number to max_heap. We negate the number because Python only has min heap
        heapq.heappush(self.max_heap, -num)

        # Move the largest number from max_heap to min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Balance the heaps
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self):
        # If heaps are of equal size, median is the average of the two middle values
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        # Else, median is the top of the larger heap
        else:
            return -self.max_heap[0]
</code></pre>

* Time complexity: O(log n) for addNum since adding a number to a heap takes logarithmic time. O(1) for findMedian, as it only involves returning the top elements of the heaps or their average.
* Space complexity: O(n), as we need to store all elements in the heaps.

---
* Follow up: If all numbers are in the range [0, 100], then we can use an array of length 101 to store the count of each number. This would allow for O(range) = O(1) insertion and a median finding. If 99% of numbers are in the range [0, 100], then we can combine the two approaches: use an array for numbers in the range [0, 100] and heaps for numbers outside this range. This optimizes for the common case while still handling outliers efficiently.

^^^

## 01 Matrix
[link]https://leetcode.com/problems/01-matrix[/link]
[tag]Medium[/tag]

Fun fact: This problem is related to what's known as the **distance transform** in computer vision and image processing. The distance transform takes a binary image and replaces each pixel with a value that represents its distance to the nearest dark pixel.

<pre><code class="language-python">class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])

        # First pass: Update the matrix from top-left to bottom-right
        for r in range(m):
            for c in range(n):
                # If the current element is not 0, calculate distances
                if mat[r][c]:
                    # Calculate the top distance, use infinity if it's the first row
                    top = mat[r - 1][c] if r > 0 else float('inf')
                    # Calculate the left distance, use infinity if it's the first column
                    left = mat[r][c - 1] if c > 0 else float('inf')
                    # Update the matrix with the minimum distance from top or left
                    mat[r][c] = min(top, left) + 1

        # Second pass: Update the matrix from bottom-right to top-left
        for r in range(m - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                # If the current element is not 0, calculate distances
                if mat[r][c]:
                    # Calculate the bottom distance, use infinity if it's the last row
                    bottom = mat[r + 1][c] if r < m - 1 else float('inf')
                    # Calculate the right distance, use infinity if it's the last column
                    right = mat[r][c + 1] if c < n - 1 else float('inf')
                    # Update the matrix with the minimum distance from bottom, right or current value
                    mat[r][c] = min(mat[r][c], bottom + 1, right + 1)

        # Return the updated matrix
        return mat
</code></pre>

* Time complexity: O(m*n). Each element in the matrix is accessed a constant number of times (twice in this case, once in each pass).
* Space complexity: O(1) as no extra space is used apart from the input matrix. All the operations are performed in-place on the given matrix.

^^^

## Spiral Matrix
[link]https://leetcode.com/problems/spiral-matrix[/link]
[tag]Medium[/tag]

<pre><code class="language-python">class Solution:
    def spiralOrder(self, matrix):
        result = []
        top, bottom, left, right = 0, len(matrix), 0, len(matrix[0])

        while top < bottom and left < right:
            # Traverse from left to right along the top row
            for i in range(left, right):
                result.append(matrix[top][i])
            top += 1

            # Traverse downwards along the rightmost column
            for i in range(top, bottom):
                result.append(matrix[i][right - 1])
            right -= 1

            # Traverse from right to left along the bottom row
            if top < bottom:
                for i in range(right - 1, left - 1, -1):
                    result.append(matrix[bottom - 1][i])
                bottom -= 1

            # Traverse upwards along the leftmost column
            if left < right:
                for i in range(bottom - 1, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1

        return result
</code></pre>

* Time Complexity: O(m * n), where m and n are the numbers of rows and columns in the matrix. Each element is visited exactly once.
* Space Complexity: O(1), if we ignore the output array. Otherwise, it's O(m * n) for the output list.

^^^

## Random Pick with Weight
[link]https://leetcode.com/problems/random-pick-with-weight/[/link]
[tag]Medium[/tag]

Fun fact: I used this algorithm in data preprocessing functions in machine learning pipelines.

<pre><code class="language-python">import random
from bisect import bisect_left

class Solution:
    def __init__(self, w):
        self.prefix_sum = [0] * len(w)
        self.prefix_sum[0] = w[0]

        # Calculate the prefix sum for each weight
        for i in range(1, len(w)):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + w[i]

    def pickIndex(self):
        # Generate a random number between 1 and the total sum of weights
        random_num = random.randint(1, self.prefix_sum[-1])
        # Use binary search to find the index of the smallest number 
        # in prefix_sum that is greater than or equal to random_num
        return bisect_left(self.prefix_sum, random_num)
</code></pre>

* Time complexity: O(N) for initialization, O(log N) for picking indices, as it uses binary search (bisect_left) on the prefix sum array.
* Space complexity: O(N) due to the storage of the prefix_sum array which has the same length as the input list.

^^^

## Kth Largest Element in an Array
[link]https://leetcode.com/problems/kth-largest-element-in-an-array/[/link]
[tag]Medium[/tag]

<pre><code class="language-python">import heapq

class Solution:
    def findKthLargest(self, nums, k):

        # Create a min-heap
        min_heap = []
        
        for num in nums:
            # Add the current number to the heap
            heapq.heappush(min_heap, num)
            
            # If the heap size exceeds k, remove the smallest element
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        
        # The root of the heap is the kth largest element
        return min_heap[0]
</code></pre>

* Time Complexity: O(N log K), where N is the number of elements in the array. For each of the N elements, we perform a heap operation that takes O(log K) time.
* Space Complexity: O(K), as we maintain a heap of size K.

^^^

## Insert Delete GetRandom O(1)
[link]https://leetcode.com/problems/insert-delete-getrandom-o1[/link]
[tag]Medium[/tag]

<pre><code class="language-python">import random

class RandomizedSet:
    def __init__(self):
        self.dict = {}  # To store elements and their indices
        self.list = []  # To store elements for random access

    def insert(self, val):
        """
        Inserts a value to the set. Returns True if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False  # Value already present
        
        self.dict[val] = len(self.list)  # Store index of val in list
        self.list.append(val)  # Append val to the list
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns True if the set contained the specified element.
        """
        if val not in self.dict:
            return False  # Value not present

        # Swap the element with the last element
        last_element = self.list[-1]
        idx = self.dict[val]
        self.list[idx], self.list[-1] = self.list[-1], self.list[idx]

        # Update the index of the last element in the dictionary
        self.dict[last_element] = idx

        # Remove the last element from list and dictionary
        self.list.pop()
        del self.dict[val]

        return True

    def getRandom(self):
        """
        Get a random element from the set.
        """
        return random.choice(self.list)
</code></pre>

* Time complexity: O(1) average time for each of the insert, remove, and getRandom operations.
* Space complexity: O(n), where n is the number of elements in the RandomizedSet, due to the storage requirements of the dictionary and the list.
