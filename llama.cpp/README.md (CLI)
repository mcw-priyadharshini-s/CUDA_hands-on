<h2>Running Models with <code>llama.cpp</code> (CUDA)</h2>

<p>
This guide explains how to build <strong><code>llama.cpp</code> with CUDA support</strong>,
convert a Hugging Face model to <strong>GGUF</strong>, and run inference using
<code>llama-cli</code>.
</p>

<hr>

<h3>1) Clone the <code>llama.cpp</code> Repository</h3>

<pre><code>git clone https://github.com/ggml-org/llama.cpp.git</code></pre>

<hr>

<h3>2) Navigate to the Repository</h3>

<pre><code>cd llama.cpp</code></pre>

<hr>

<h3>3) Build <code>llama-cli</code> Using CMake (CUDA Enabled)</h3>

<pre><code>mkdir -p build
cd build
cmake .. -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j</code></pre>

<p>
⚠️ <strong>Note:</strong><br>
Use a normal hyphen (<code>-j</code>), not an en dash (<code>–j</code>),
or CMake will fail.
</p>

<hr>

<h3>4) Locate the Executable</h3>

<p>
After a successful build, <code>llama-cli</code> will be available at:
</p>

<pre><code>build/bin/llama-cli</code></pre>

<hr>

<h3>5) Convert a Hugging Face Model to GGUF (If Needed)</h3>

<p>
If the model already provides a <code>.gguf</code> file, this step can be skipped.
Otherwise, use the conversion script provided by <code>llama.cpp</code>.
</p>

<h4>Example: Converting <em>Phi-3.5-mini-instruct</em> to FP16 GGUF</h4>

<pre><code>python3 convert_hf_to_gguf.py microsoft/Phi-3.5-mini-instruct \
  --remote \
  --verbose \
  --outfile ./models/Microsoft-phi-3.5-f16 \
  --outtype f16</code></pre>

<p>
📁 <strong>Note:</strong> <code>models/</code> is the directory where the generated
<code>.gguf</code> file will be stored.
</p>

<hr>

<h3>6) Run Inference Using <code>llama-cli</code></h3>

<p>
If multiple prompts are used, they can be placed in a text file
(one prompt per line), for example <code>prompts.txt</code>.
</p>

<h4>Example Command</h4>

<pre><code>./build/bin/llama-cli \
  --model &lt;path_to_model.gguf&gt; \
  -f prompts.txt \
  -b 2 \
  -n 200 \
  -c 2048 \
  --perf</code></pre>

<hr>

<h3>Notes on Performance</h3>

<ul>
  <li>This model performs well with the parameters shown above.</li>
  <li>Additional options such as:</li>
  <ul>
    <li><code>--flash-attn</code></li>
    <li><code>--cache-type-k</code></li>
    <li><code>--cache-type-v</code></li>
  </ul>
  <li>
    These generally show <strong>negligible throughput differences</strong>
    for this model.
  </li>
</ul>

<hr>

<h3>References</h3>

<ul>
  <li>
    <strong><code>llama.cpp</code></strong>:
    https://github.com/ggml-org/llama.cpp
  </li>
  <li>
    GGUF format documentation is available in the <code>llama.cpp</code> repository.
  </li>
</ul>
