def solve_sudoku(grid_str: str) -> str | None:
    board = [int(c) for c in grid_str]

    # bitmask untuk row, col, box
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9

    empties = []

    # init
    for i in range(81):
        if board[i] == 0:
            empties.append(i)
        else:
            r, c = divmod(i, 9)
            b = (r // 3) * 3 + (c // 3)
            mask = 1 << board[i]
            row[r] |= mask
            col[c] |= mask
            box[b] |= mask

    def get_candidates(pos):
        r, c = divmod(pos, 9)
        b = (r // 3) * 3 + (c // 3)
        used = row[r] | col[c] | box[b]
        return (~used) & 0x3FE  # bit 1-9

    def backtrack():
        if not empties:
            return True

        # 🔥 MRV: pilih cell dengan kandidat paling sedikit
        min_i = -1
        min_count = 10

        for i in range(len(empties)):
            pos = empties[i]
            mask = get_candidates(pos)
            count = bin(mask).count("1")
            if count < min_count:
                min_count = count
                min_i = i
            if count == 1:
                break

        if min_count == 0:
            return False

        pos = empties.pop(min_i)
        r, c = divmod(pos, 9)
        b = (r // 3) * 3 + (c // 3)

        mask = get_candidates(pos)

        while mask:
            num = (mask & -mask)  # ambil bit paling kanan
            digit = num.bit_length() - 1

            # place
            board[pos] = digit
            row[r] |= num
            col[c] |= num
            box[b] |= num

            if backtrack():
                return True

            # undo
            board[pos] = 0
            row[r] ^= num
            col[c] ^= num
            box[b] ^= num

            mask &= mask - 1  # hapus bit

        empties.insert(min_i, pos)
        return False

    if backtrack():
        return "".join(map(str, board))
    return None

def is_valid_board(board):
    for i in range(9):
        row = set()
        col = set()
        box = set()
        for j in range(9):
            # row
            if board[i*9 + j] != 0:
                if board[i*9 + j] in row:
                    return False
                row.add(board[i*9 + j])

            # col
            if board[j*9 + i] != 0:
                if board[j*9 + i] in col:
                    return False
                col.add(board[j*9 + i])

            # box
            r = (i // 3) * 3 + j // 3
            c = (i % 3) * 3 + j % 3
            if board[r*9 + c] != 0:
                if board[r*9 + c] in box:
                    return False
                box.add(board[r*9 + c])
    return True

def format_sudoku(grid_str: str) -> str:
    lines = []
    for i in range(0, 81, 9):
        row = grid_str[i:i+9]
        lines.append(f"{row[0:3]} {row[3:6]} {row[6:9]}")
    return "\n".join(lines)
