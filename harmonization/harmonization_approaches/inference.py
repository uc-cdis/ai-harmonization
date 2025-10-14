import pandas as pd
from tqdm import trange
from vllm.lora.request import LoRARequest


def get_df_for_batch_outputs_from_llm(
    test_df, lo, N, batch_size, llm, lora_model_path, sampling_params
):

    hi = lo + batch_size
    if hi > N:
        hi = N

    temp_df_for_batch = pd.DataFrame(
        columns=["label", "input_text", "expected_output_text", "generated_output_text"]
    )

    batch_labels = test_df.iloc[lo:hi].get("label")
    batch_inputs = test_df.iloc[lo:hi]["input_text"]
    true_outputs = test_df.iloc[lo:hi]["expected_output_text"]
    batch_outputs = llm.generate(
        batch_inputs,
        sampling_params,
        lora_request=LoRARequest("arpa-h-eval", 1, lora_model_path),
    )
    batch_outputs = [o.outputs[0].text for o in batch_outputs]

    temp_df_for_batch["label"] = batch_labels
    temp_df_for_batch["input_text"] = batch_inputs
    temp_df_for_batch["expected_output_text"] = true_outputs
    temp_df_for_batch["generated_output_text"] = batch_outputs

    return temp_df_for_batch


def batch_inference(test_df, batch_size, llm, lora_model_path, sampling_params):

    results_df = pd.DataFrame(
        columns=["label", "input_text", "expected_output_text", "generated_output_text"]
    )
    N = len(test_df)

    for lo in trange(0, N, batch_size):
        temp_df_for_batch = get_df_for_batch_outputs_from_llm(
            test_df=test_df,
            lo=lo,
            N=N,
            batch_size=batch_size,
            llm=llm,
            lora_model_path=lora_model_path,
            sampling_params=sampling_params,
        )
        results_df = pd.concat([results_df, temp_df_for_batch], axis=0)

    return results_df
