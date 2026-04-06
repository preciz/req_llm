defmodule ReqLLM.Providers.MinimaxTest do
  @moduledoc """
  Provider-level tests for Minimax implementation.

  Tests the provider contract, configuration, and OpenAI-compatible
  request/response handling without making live API calls.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Minimax

  alias ReqLLM.Providers.Minimax

  defp minimax_model(model_id \\ "minimax-text-01", opts \\ []) do
    %LLMDB.Model{
      id: "minimax:#{model_id}",
      model: model_id,
      name: Keyword.get(opts, :name, "Minimax Test Model"),
      provider: :minimax,
      family: Keyword.get(opts, :family, "test"),
      capabilities: Keyword.get(opts, :capabilities, %{chat: true, tools: %{enabled: true}}),
      limits: Keyword.get(opts, :limits, %{context: 32_000, output: 4096})
    }
  end

  describe "provider contract" do
    test "provider identity and configuration" do
      assert Minimax.provider_id() == :minimax
      assert is_binary(Minimax.base_url())
      assert Minimax.base_url() == "https://api.minimax.io/v1"
    end

    test "provider uses correct default environment key" do
      assert Minimax.default_env_key() == "MINIMAX_API_KEY"
    end

    test "provider schema contains response_format" do
      schema_keys = Minimax.provider_schema().schema |> Keyword.keys()
      assert :response_format in schema_keys
    end
  end

  describe "request preparation" do
    test "prepare_request for :chat creates /chat/completions request" do
      model = minimax_model()
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Minimax.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/chat/completions"
      assert request.method == :post
    end

    test "prepare_request for :object injects json_schema instruction and response_format" do
      model = minimax_model()
      prompt = %ReqLLM.Context{messages: [%ReqLLM.Message{role: :user, content: [%ReqLLM.Message.ContentPart{type: :text, text: "Hello"}]}]}
      
      schema = %{
        type: "object",
        properties: %{name: %{type: "string"}},
        required: ["name"]
      }
      {:ok, compiled_schema} = ReqLLM.Schema.compile(schema)

      opts = [
        compiled_schema: compiled_schema,
        tools: [ReqLLM.Tool.new!(name: "test", description: "d", parameter_schema: %{}, callback: fn _ -> :ok end)],
        tool_choice: %{type: "function", function: %{name: "test"}}
      ]

      {:ok, request} = Minimax.prepare_request(:object, model, prompt, opts)

      assert %Req.Request{} = request
      
      # Tools should be removed
      assert request.options[:tools] == nil
      assert request.options[:tool_choice] == nil
      
      # Max tokens set
      assert request.options[:max_tokens] == 4096
      
      # Operation set
      assert request.options[:operation] == :object
      
      # response_format added to provider_options
      provider_opts = request.options[:provider_options] || []
      assert provider_opts[:response_format][:type] == "json_schema"
      
      # System instruction injected
      context = request.options[:context]
      assert length(context.messages) == 2
      system_message = hd(context.messages)
      assert system_message.role == :system
      assert String.contains?(system_message.content |> hd() |> Map.get(:text), "ONLY valid JSON matching this schema")
    end
  end
end
