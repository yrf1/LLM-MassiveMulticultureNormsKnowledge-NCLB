"""
Chat with a model with command line interface.

Setup:
    python3 -m venv Vicuna_env

Usage:
python eval/benchmark_Vicuna.py --model lmsys/vicuna-7b-v1.3
python eval/benchmark_Vicuna.py --model lmsys/vicuna-13b-v1.3 --num-gpus 2
python eval/benchmark_Vicuna.py --model lmsys/vicuna-33b-v1.3 --num-gpus 2
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
"""
import argparse
import os
import re
import sys
import json
import random
import pandas as pd

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.inference import ChatIO, chat_loop
from fastchat.model.model_adapter import load_model

from datetime import datetime

sys.path.append("eval")
from utils import read_computed_stats

class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        #print(type(output_stream), "aa")
        logits = []
        for outputs in output_stream:
            logits.append(outputs["token_logit"])
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        #print(" ".join(output_text[pre:]), flush=True)
        return logits, " ".join(output_text)


class RichChatIO(ChatIO):
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset"], pattern=re.compile("$")
        )
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        contents = ""
        # `end_sequence` signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = " __END_OF_A_MESSAGE_47582648__\n"
        len_end = len(end_sequence)
        while True:
            if len(contents) >= len_end:
                last_chars = contents[-len_end:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        contents = contents[:-len_end]
        print(f"[!OP:{role}]: {contents}", flush=True)
        return contents

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

def read_from_json_per_c_t(input_dir, country, topic):
    """
    Old/Deprecated? Needs to be Updated
    """
    result_fname = input_dir+"/"+country+"/"+topic+"/1_output_"+ckpt_dir.replace("/","")+".json"

    if self_mode == "evaluate":
        if os.path.exists(result_fname):
            return None, None

        try:
            with open(input_dir+"/"+country+"/"+topic+"/1.json", "r") as f:
                input_s_list = json.load(f)
        except:
            print("Cannot load and skipping... "+input_dir+"/"+country+"/"+topic+"/1.json")
            return None, None

        results = []
        for x in input_s_list:
            try:
                pred_output = run_eval_per_s_query(x)
                output_s_list.append((query_s, \
                   pred_output['generation']['content']))
            except:
                pass
    return output_s_list, result_fname

def read_from_csv_agg(input_dir, result_fname, generator, tokenizer=None, args=None):
    """
    New, Latest
    """
    df = pd.read_csv(input_dir+"benchmarK_Feb4_neg10k.csv")
    output_s_list = []

    for idx, data in df.iterrows():
        c, t1, tit, lbl = data[0], data[1], data[4], data[10]
        query_s = data[9]
        if True: #try:
            if type(query_s) is float:
                continue
            if "[NEG NORM]" in query_s:
                query_s = query_s.split("[NEG NORM]")[0]
            # TODO: fix/update this
            _, pred_output = main(args, query_s + " True or False:", generator, tokenizer)
            #pred_output = run_eval_per_s_query(query_s, generator)
            pred_output = pred_output.split("ASSISTANT: ")[-1]
            output_s_list.append((c, t1, tit, query_s, lbl, \
                           pred_output))
        #except:
        #    pass
        if idx % 100 == 0:
            print("~~~~", idx, df.shape[0], datetime.now())
            with open(result_fname, "w") as f:
                json.dump(output_s_list, f)
    return output_s_list

def main(args, input_s, model, tokenizer):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if args.style == "simple":  # Yi: this is default (what we dealing with)
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    if True: #try:
        s = chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            args.revision,
            args.judge_sent_end,
            args.debug,
            history=not args.no_history,
            inp=input_s, 
            model=model,
            tokenizer=tokenizer
        )
        out_s = chatio.stream_output(s)
        return out_s
    #except KeyboardInterrupt:
    #    print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()

    self_mode = ['evaluate','read computed stats'][0] 
    ablation = "none" #low_resource"
    
    if self_mode == "evaluate":
        model, tokenizer = load_model(
            args.model_path,
            args.device, args.num_gpus, args.max_gpu_memory,
            args.load_8bit, args.cpu_offloading,
            GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ), args.revision, args.debug)
    else:
        model, tokenizer = None, None

    """
    input_s_list_labels = [True, False, True, False, True, True, False]
    input_s_list = ["HK is part of China, true or false"]
    input_s_list = ["In Chinese culture, people give red envelopes to most people that they know and pass by during lunar new year.", \
        "In Northern Chinese culture, people give red envelopes to most people that they know and pass by during lunar new year.", \
        #"In Southern Chinese culture, people give red envelopes to most people that they know and pass by during lunar new year.", \
        #"In American culture, people often wear uniforms to high school.", \
        #"In American culture, people may often wear uniforms going to private high school.", \
        #"In Philippines culture, it is typically expected to take off shoes when entering someone's home.", \
        "Lychee is a signature fruit of Dongbei China."]
    output_fname = "Vicuna13b_sanity_check.json"
    """
    input_dir = "data/culture_scraped_info/" 

    model_name = args.model_path.split("/")[-1].split("-v1")[0]
    input_dir = input_dir.split("benchmark/")[0]
    result_fname = input_dir + "benchmark_Feb4_neg_"+model_name+"_eval_result_agg.json"

    if self_mode == "evaluate":
        start_t = datetime.now()
        output_s_list = read_from_csv_agg(input_dir, result_fname, model, tokenizer, args)
 
    if self_mode == "read computed stats":
        read_computed_stats(result_fname, model_name)
