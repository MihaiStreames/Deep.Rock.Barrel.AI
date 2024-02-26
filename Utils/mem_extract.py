from Utils.game_attrs import PTR_DICT
from pymem import Pymem
from pymem.process import module_from_name

### Imports ###

class MemExtract:
    def __init__(self, executable: str):
        self.pm = Pymem(executable)
        self.game_module = module_from_name(self.pm.process_handle, executable).lpBaseOfDll

    def get_pointer_address(self, base: hex, offsets: list[hex]) -> hex:
        addr = self.pm.read_longlong(base)

        for i in offsets:
            if i is not offsets[-1]:
                addr = self.pm.read_longlong(addr + i)
            else:
                return addr + offsets[-1]

    def extract_memory(self):
        memory_data = {
            'score': None,
            'kicks': None,
        }

        # Extract score
        score_address = self.get_pointer_address(self.game_module + PTR_DICT['score']['base'], PTR_DICT['score']['offsets'])
        memory_data['score'] = self.pm.read_longlong(score_address)

        # Extract kicks
        kicks_address = self.get_pointer_address(self.game_module + PTR_DICT['kicks']['base'], PTR_DICT['kicks']['offsets'])
        memory_data['kicks'] = self.pm.read_int(kicks_address)

        return memory_data